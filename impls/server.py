#!/usr/bin/env python3
import os
import socket
import pickle
import argparse
import time
import inspect
import json
from pathlib import Path
from collections import deque
import importlib

import numpy as np
import jax
import jax.numpy as jnp
from flax.serialization import from_bytes, to_state_dict, from_state_dict, msgpack_restore
import optax
import ml_collections

# ---------------- XLA memory knobs (same as your setup) ----------------
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

# ---------------- util: import agent family ----------------
def build_agent(agent_str: str):
    module = importlib.import_module(f"agents.{agent_str}")
    Agent = getattr(module, f"{agent_str.upper()}Agent")
    get_config = getattr(module, "get_config")
    return Agent, get_config

# ---------------- socket helpers ----------------
def send_len_pickled(conn: socket.socket, obj):
    """Send one header: [4-byte big-endian length] + pickle(obj)."""
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(len(payload).to_bytes(4, "big") + payload)

def recvall(conn: socket.socket, n: int, *, timeout: float | None = None) -> bytes:
    if timeout is not None:
        conn.settimeout(timeout)
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection lost while receiving payload")
        buf += chunk
    return buf

# ---------------- shape helper (no cv2) ----------------
def fit_to_hw(img: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Center-crop to (H,W) if larger; pad with edge pixels if smaller.
    Keeps channels untouched. No resampling libs needed.
    """
    h, w = img.shape[:2]
    y0 = max(0, (h - H) // 2)
    x0 = max(0, (w - W) // 2)
    # crop
    cropped = img[y0:min(y0 + H, h), x0:min(x0 + W, w)]
    ch = cropped.shape[2] if cropped.ndim == 3 else 1
    # pad if needed
    pad_h = H - cropped.shape[0]
    pad_w = W - cropped.shape[1]
    if pad_h > 0 or pad_w > 0:
        if cropped.ndim == 2:
            cropped = cropped[:, :, None]
        cropped = np.pad(
            cropped,
            ((0, max(0, pad_h)), (0, max(0, pad_w)), (0, 0)),
            mode="edge",
        )
        cropped = cropped[:H, :W, :]
    return cropped

# ---------------- main ----------------
def main():
    p = argparse.ArgumentParser()
    # Server-specific arguments
    p.add_argument("--agent", default="gcbc", choices=["crl", "cmd", "gcbc", "gciql", "tmd"])
    p.add_argument("--model_path",
                   default="/global/scratch/users/achyuthkv76/tmd_models/run2.pkl")
    p.add_argument("--config_path", type=str, default=None,
                   help="Path to config.json file (saved during training). If provided, agent will be created based on this config.")
    p.add_argument("--dataset_path",
                   default="/nfs/kun2/users/achyuth/carla_test_scripts/goals.npz")
    p.add_argument("--goal_frame_index", type=int, default=1000)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=5050)
    
    # Match train.py arguments EXACTLY (same names, same defaults, same help text)
    # Note: Training resizes to 64x64x3, so server should match that
    p.add_argument("--obs_h", type=int, default=64)
    p.add_argument("--obs_w", type=int, default=64)
    p.add_argument("--obs_c", type=int, default=3)
    p.add_argument("--frame_offsets", nargs="*", type=int, default=None, help="e.g., --frame_offsets 0 -5 -10 -20")
    p.add_argument("--block_size", type=int, default=400, help="Block size for block-aware frame stacking and shuffling")
    p.add_argument("--action_chunk_length", type=int, default=1, help="Number of actions to predict in sequence (1 = disabled, typical: 4-16)")
    p.add_argument("--n_actions", type=int, default=4, help="Use only the first N actions from the chunk (default: 4 = use first 4 actions). Useful when model was trained with chunking but you want to execute only the first N actions.")
    p.add_argument("--use_discrete", action="store_true", default=False, help="Use discrete actions (multi-discrete mode: discretize throttle/steer/brake into 32 bins each)")
    args = p.parse_args()

    # ---- Load agent ----
    agent_str = args.agent.lower()
    module = importlib.import_module(f"agents.{agent_str}")
    
    Agent = getattr(module, f"{agent_str.upper()}Agent")
    get_config = getattr(module, "get_config")

    # Load config from file if provided, otherwise use defaults
    config = get_config()
    if args.config_path:
        config_path = Path(args.config_path)
        if config_path.exists():
            print(f"Loading config from {config_path}")
            with config_path.open("r") as f:
                config_dict = json.load(f)
            # Update config with values from file
            for key, value in config_dict.items():
                # Handle tuple/list conversion for fields like frame_offsets, hidden_dims, etc.
                if isinstance(value, list) and key in [
                    "frame_offsets",
                    "actor_hidden_dims",
                    "value_hidden_dims",
                    "distance_head_hidden_dims",
                ]:
                    value = tuple(value)
                # For ml_collections.ConfigDict, allow creating new fields directly
                if isinstance(config, ml_collections.ConfigDict):
                    config[key] = value
                else:
                    setattr(config, key, value)
            print(f"Loaded config from file: {len(config_dict)} keys")
        else:
            print(f"Warning: config file {config_path} not found, using defaults")
    else:
        print("No config file provided, using defaults from agent.get_config()")

    obs_shape = (args.obs_h, args.obs_w, args.obs_c)
    frame_stack_k = int(config.get('frame_stack', len(config.frame_offsets)))
    stacked_obs_shape = (args.obs_h, args.obs_w, args.obs_c * frame_stack_k)
    act_shape = (3,)
    action_dim = act_shape[0]  # 3

    # Load checkpoint bytes.
    # New-style checkpoints (train.py) save pure Flax bytes via fxs.to_bytes(agent),
    # so we read raw bytes. For backwards compatibility, if this somehow loads a
    # pickled dict, fall back gracefully.
    with open(args.model_path, "rb") as f:
        raw = f.read()
    try:
        # If it's actually a pickled dict, handle the old format.
        maybe_dict = pickle.loads(raw)
        if isinstance(maybe_dict, dict) and "agent" in maybe_dict:
            checkpoint_bytes = maybe_dict["agent"]
            # Optional: config fallback from checkpoint if no config_path
            if not args.config_path and "config" in maybe_dict:
                saved_config = maybe_dict["config"]
                if isinstance(saved_config, dict):
                    for key, value in saved_config.items():
                        if isinstance(value, list) and key in [
                            "frame_offsets",
                            "actor_hidden_dims",
                            "value_hidden_dims",
                            "distance_head_hidden_dims",
                        ]:
                            value = tuple(value)
                        if isinstance(config, ml_collections.ConfigDict):
                            config[key] = value
                        else:
                            setattr(config, key, value)
                    print(f"[INFO] Loaded config from checkpoint (fallback): {len(saved_config)} keys")
        else:
            # Not a dict; assume raw is already Flax bytes
            checkpoint_bytes = raw
    except Exception:
        # Not a pickle; assume raw is Flax bytes (current format)
        checkpoint_bytes = raw
    
    # Initialize with *stacked* shape (this must match runtime)
    dummy_obs = jnp.zeros((1, *stacked_obs_shape), dtype=jnp.float32)
    dummy_act = jnp.zeros((1, *act_shape), dtype=jnp.float32)  # (1, 3)
    
    # Create agent with config that matches checkpoint (structure will match)
    agent = Agent.create(
        seed=0,
        ex_observations=dummy_obs,
        ex_actions=dummy_act,
        config=config,
    )
    
    # Load checkpoint state dict directly (bypasses agent structure validation)
    # Extract only params to avoid opt_state structure mismatches
    checkpoint_state_dict = msgpack_restore(checkpoint_bytes)
    current_state = to_state_dict(agent)
    
    # Copy only params from checkpoint, keep fresh opt_state from current agent
    if 'network' in checkpoint_state_dict and 'params' in checkpoint_state_dict['network']:
        current_state['network']['params'] = checkpoint_state_dict['network']['params']
        # Optionally preserve step if it exists
        if 'network' in checkpoint_state_dict and 'step' in checkpoint_state_dict['network']:
            current_state['network']['step'] = checkpoint_state_dict['network']['step']
        agent = from_state_dict(agent, current_state)
    else:
        raise ValueError("Could not find network params in checkpoint")
    
    print("Model loaded and agent initialized")
    
    print(args.goal_frame_index)
    # ---- Load dataset & goal (resize to match training: 64x64x3) ----
    dataset = np.load(args.dataset_path)
    frames = dataset["frames"]
    num_frames = len(frames)
    
    # Stack goal frames using contiguous history, oldest-first (matches GCDataset).
    # Goal at index g gets frames [g-(k-1), ..., g-1, g], clamped to trajectory start.
    goal_frame_stack = []
    for j in range(frame_stack_k):
        goal_idx = args.goal_frame_index - (frame_stack_k - 1 - j)
        goal_idx = max(0, min(goal_idx, num_frames - 1))
        gi_raw = np.asarray(frames[goal_idx])
        gi_frame = fit_to_hw(gi_raw, args.obs_h, args.obs_w)
        if gi_frame.dtype != np.uint8:
            if gi_frame.max() <= 1.0:
                gi_frame = (gi_frame * 255.0).astype(np.uint8)
            else:
                gi_frame = np.clip(gi_frame, 0, 255).astype(np.uint8)
        goal_frame_stack.append(gi_frame)

    goal_stacked = np.concatenate(goal_frame_stack, axis=-1)  # (H, W, 3*k) uint8
    gi_obs = goal_frame_stack[-1]  # current frame for display

    # optional goal (x,y)
    goal_xy = None
    for k in ("goal_xy", "poses_xy", "poses", "goal_locs", "loc_xy", "xy"):
        if k in dataset:
            arr = np.asarray(dataset[k][args.goal_frame_index])
            if arr.size >= 2:
                goal_xy = (float(arr[0]), float(arr[1]))
                break

    # Load goal for header (use first frame for display)
    goal_fixed_header = gi_obs  # uint8 (H,W,3) for header display only

    print("Goal loaded")
    # ---- Socket setup ----
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.host, args.port))
    server_sock.listen(1)
    print(f"Listening on {args.host}:{args.port}...")
    conn, addr = server_sock.accept()
    print(f"Connection established with client: {addr}")

    # send one-time header: goal image + (x,y), but using ONLY plain Python
    # types (no numpy objects) to avoid pickle depending on numpy internals
    # on the client side.
    try:
        gi_u8 = np.asarray(gi_obs, dtype=np.uint8)
        header = {
            "goal_img": {
                "shape": list(gi_u8.shape),
                "dtype": "uint8",
                "data": gi_u8.reshape(-1).tolist(),
            },
            "goal_xy": goal_xy,
        }
        send_len_pickled(conn, header)
        print(f"Sent goal header (img {gi_u8.shape}, xy={goal_xy})")
    except Exception as e:
        print(f"[WARN] failed to send goal header: {e}")

    # ---- Runtime buffers ----
    # Match GCDataset: contiguous frame stack of length frame_stack.
    # Oldest-first: [t-(k-1), ..., t-1, t] concatenated along channels.
    frame_stack_k = int(config.get('frame_stack', len(config.frame_offsets)))
    HISTORY = deque(maxlen=frame_stack_k)
    
    # Direct module access (matching your previous working pattern)
    actor_module = agent.network.model_def.modules["actor"]
    actor_params = agent.network.params["modules_actor"]

    print("Runtime config:")
    print(f"  obs_shape={obs_shape}")
    print(f"  stacked_obs_shape={stacked_obs_shape}")
    print(f"  frame_stack={frame_stack_k}")
    
    # Goal must be uint8 [0,255] — same as training observations. ImpalaEncoder normalises.
    goal_fixed = jnp.array(goal_stacked[None, ...])  # (1, H, W, 3*k) uint8
    
    t0 = time.perf_counter()
    
    try:
        while True:
            # 1) receive image from client
            length_bytes = conn.recv(4)
            if not length_bytes:
                break
            msg_len = int.from_bytes(length_bytes, "big")
            data = recvall(conn, msg_len, timeout=30.0)
            t_recv = time.perf_counter()

            # 2) deserialize; resize to (obs_h, obs_w, 3) to match training
            img = pickle.loads(data)
            img_arr = np.asarray(img)

            if img_arr.shape[:2] != (args.obs_h, args.obs_w):
                img_arr = fit_to_hw(img_arr, args.obs_h, args.obs_w)

            # Ensure uint8 [0,255] — ImpalaEncoder does /255.0 internally.
            # Training stores uint8 and passes directly; we must do the same.
            if img_arr.dtype != np.uint8:
                if img_arr.max() <= 1.0:
                    img_arr = np.clip(img_arr * 255.0, 0, 255).astype(np.uint8)
                else:
                    img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)

            obs = jnp.array(img_arr).reshape((1, *obs_shape))  # (1,H,W,3) uint8

            HISTORY.append(obs)
            # Contiguous frame stack, oldest-first (matches GCDataset.get_stacked_observations).
            # If we have fewer than k frames, repeat the oldest available.
            obs_stack = []
            n = len(HISTORY)
            for j in range(frame_stack_k):
                idx = max(0, n - frame_stack_k + j)
                obs_stack.append(HISTORY[idx])

            obs_stacked = jnp.concatenate(obs_stack, axis=-1)

            # ---- match your previously working call pattern ----
            obs_fixed = obs_stacked.copy()
            # goal_fixed is already prepared before the loop

            # DEBUG: Print observation mean to check if it's changing
            obs_mean = float(np.array(obs_fixed).mean())
            print(f"[DEBUG step {n}] obs mean={obs_mean:.6f}")

            # DEBUG: Log input shapes and ranges to match training
            if n == 1:  # Only log on first iteration to avoid spam
                print(f"\n[DEBUG] Input shapes and ranges (matching training format):")
                print(f"  obs_fixed: shape={obs_fixed.shape}, dtype={obs_fixed.dtype}, range=[{obs_fixed.min():.3f}, {obs_fixed.max():.3f}]")
                print(f"  goal_fixed: shape={goal_fixed.shape}, dtype={goal_fixed.dtype}, range=[{goal_fixed.min():.3f}, {goal_fixed.max():.3f}]")

            # 4) forward (explicit kwargs to avoid signature mismatches)
            t_prep = time.perf_counter()
            # Flax apply() requires variables as first positional argument
            variables = {"params": actor_params}
            apply_kwargs = {
                "observations": obs_fixed,
                "goals": goal_fixed,
                "goal_encoded": False,
                "temperature": 1.0,
            }
            
            action_dist = actor_module.apply(variables, **apply_kwargs)
            act = action_dist.mean()       # (1, action_dim * chunk_len) flat
            t_fwd = time.perf_counter()

            # Reshape flat output to (chunk_len, action_dim)
            chunk_len = int(config.get('action_chunk_length', 1))
            act_np = np.array(act[0], dtype=np.float32)  # (action_dim * chunk_len,)
            if chunk_len > 1:
                act_chunk = act_np.reshape(chunk_len, -1)  # (chunk_len, action_dim)
            else:
                act_chunk = act_np.reshape(1, -1)           # (1, action_dim)
            chunk_length = act_chunk.shape[0]
            
            # Optionally use only the first N actions from the chunk
            if args.n_actions is not None and chunk_length > args.n_actions:
                original_chunk_length = chunk_length
                act_chunk = act_chunk[:args.n_actions]
                chunk_length = args.n_actions
                print(f"[INFO] Using only first {chunk_length} actions from chunk (model predicted {original_chunk_length} actions)")
            
            # Ensure actions are in valid range [0,1] for throttle/brake, [-1,1] for steer
            act_chunk[:, 0] = np.clip(act_chunk[:, 0], 0.0, 1.0)  # throttle
            act_chunk[:, 1] = np.clip(act_chunk[:, 1], -1.0, 1.0)  # steer
            act_chunk[:, 2] = np.clip(act_chunk[:, 2], 0.0, 1.0)  # brake
            
            # Send chunk: [4-byte chunk_length] + [chunk_length * 3 * float32]
            chunk_length_bytes = chunk_length.to_bytes(4, "big")
            action_bytes = act_chunk.flatten().tobytes()  # Flatten to (chunk_length * 3) float32s
            print(f"Sending action chunk: shape={act_chunk.shape}, chunk_length={chunk_length}, discrete={config.get('discrete', False)}")
            conn.sendall(chunk_length_bytes + action_bytes)
            t_send = time.perf_counter()

            # Optional perf:
            # print(f"recv={t_recv-t0:.3f} prep={t_prep-t_recv:.3f} fwd={t_fwd-t_prep:.3f} send={t_send-t_fwd:.3f}")
            t0 = time.perf_counter()

    except KeyboardInterrupt:
        print("Interrupted by User")
    finally:
        try: conn.close()
        except Exception: pass
        try: server_sock.close()
        except Exception: pass
        print("Server shutdown cleanly")

if __name__ == "__main__":
    main()
