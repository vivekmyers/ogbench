#!/usr/bin/env python3
import os
import socket
import pickle
import argparse
import time
from collections import deque
import importlib

import numpy as np
import jax
import jax.numpy as jnp
from flax.serialization import from_bytes

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
    p.add_argument("--agent", default="gcbc", choices=["crl", "cmd", "gcbc", "gciql"])
    p.add_argument("--model_path",
                   default="/global/scratch/users/achyuthkv76/crl_models/run4.pkl")
    p.add_argument("--dataset_path",
                   default="/global/scratch/users/achyuthkv76/carla_test_scripts/goals_1024.npz")
    p.add_argument("--goal_frame_index", type=int, default=1)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=5050)
    p.add_argument("--network_dim", type=int, default=512)
    p.add_argument("--latent_dim", type=int, default=512)

    # keep 100x100 as per your training
    p.add_argument("--obs_h", type=int, default=100)
    p.add_argument("--obs_w", type=int, default=100)
    p.add_argument("--obs_c", type=int, default=3)
    p.add_argument("--frame_stack", type=int, default=4)
    args = p.parse_args()

    # ---- Load agent ----
    agent_str = args.agent.lower()
    module = importlib.import_module(f"agents.{agent_str}")
    
    Agent = getattr(module, f"{agent_str.upper()}Agent")
    get_config = getattr(module, "get_config")

    config = get_config()
    config.encoder = "impala_large"
    config.actor_loss = "ddpgbc"
    config.discrete = False
    config.multi_discrete = False
    config.frame_stack = args.frame_stack
    config.actor_hidden_dims = (args.network_dim, args.network_dim, args.network_dim)
    config.value_hidden_dims = (args.network_dim, args.network_dim, args.network_dim)
    config.latent_dim = args.latent_dim
    config.layer_norm = False
    config.frame_offsets = [0, -10, -20, -50, -80]

    obs_shape = (args.obs_h, args.obs_w, args.obs_c)                     # (100,100,3)
    # Use frame_offsets to determine actual stacked shape, not frame_stack
    num_frame_offsets = len(config.frame_offsets)
    stacked_obs_shape = (args.obs_h, args.obs_w, args.obs_c*num_frame_offsets)  # (100,100,15) with 5 offsets
    act_shape = (3,)

    # Initialize with *stacked* shape (this must match runtime)
    dummy_obs = jnp.zeros((1, *stacked_obs_shape), dtype=jnp.float32)
    dummy_act = jnp.zeros((1, *act_shape), dtype=jnp.float32)
    agent = Agent.create(seed=0, ex_observations=dummy_obs, ex_actions=dummy_act, config=config)

    with open(args.model_path, "rb") as f:
        agent = from_bytes(agent, f.read())
    print("Model loaded and agent initialized")
    print(args.goal_frame_index)
    # ---- Load dataset & goal (stay at 100x100; crop/pad if needed) ----
    dataset = np.load(args.dataset_path)
    gi_raw = np.asarray(dataset["frames"][args.goal_frame_index])  # possibly already (100,100,3)
    gi_obs = fit_to_hw(gi_raw, args.obs_h, args.obs_w).astype(np.float32)  # (100,100,3)

    # optional goal (x,y)
    goal_xy = None
    for k in ("goal_xy", "poses_xy", "poses", "goal_locs", "loc_xy", "xy"):
        if k in dataset:
            arr = np.asarray(dataset[k][args.goal_frame_index])
            if arr.size >= 2:
                goal_xy = (float(arr[0]), float(arr[1]))
                break

    # stack goal along channels to match network's goal branch (if used)
    goal = gi_obs
    if np.max(gi_obs) > 1.0:
        goal = goal / 255.0
    GOAL_HISTORY = deque(maxlen=(config.frame_offsets[-1] * (-1) + 1))
    GOAL_HISTORY.append(goal)
    g = len(GOAL_HISTORY)
    raw = -1 + np.array(config.frame_offsets, dtype=int)
    raw = np.maximum(-g, raw)
    idx = [(g + i) if i < 0 else int(i) for i in raw.tolist()]
    goal_stack = [GOAL_HISTORY[k] / 255.0 for k in idx]
    goal_stacked = jnp.concatenate(goal_stack, axis=-1)
    goal_fixed = []
    goal_fixed.append(goal_stacked.copy())
    goal_fixed = jnp.array(goal_fixed)

    print("Goal loaded")
    # ---- Socket setup ----
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.host, args.port))
    server_sock.listen(1)
    print(f"Listening on {args.host}:{args.port}...")
    conn, addr = server_sock.accept()
    print(f"Connection established with client: {addr}")

    # send one-time header: goal image (uint8, 100x100x3) + (x,y)
    try:
        gi_send = gi_obs
        # convert float32 [0..1] or [0..255] into uint8 safely
        mx = float(np.max(gi_send)) if gi_send.size else 1.0
        if mx <= 1.0:
            gi_u8 = np.clip(gi_send, 0.0, 1.0)*255.0
        else:
            gi_u8 = np.clip(gi_send, 0.0, 255.0)
        gi_u8 = (gi_u8 + 0.5).astype(np.uint8)
        header = {"goal_img": gi_u8, "goal_xy": goal_xy}
        send_len_pickled(conn, header)
        print(f"Sent goal header (img {gi_u8.shape}, xy={goal_xy})")
    except Exception as e:
        print(f"[WARN] failed to send goal header: {e}")

    # ---- Runtime buffers ----
    FRAME_STACK = config.frame_stack
    obs_stack = deque(maxlen=FRAME_STACK)
    HISTORY = deque(maxlen=(config.frame_offsets[-1] * (-1) + 1))
    actor_module = agent.network.model_def.modules["actor"]
    actor_params = agent.network.params["modules_actor"]

    print(f"Runtime obs={obs_shape}, stacked={stacked_obs_shape}, stack={FRAME_STACK}")
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

            # 2) deserialize; ensure (100,100,3) via crop/pad (no cv2)
            img = pickle.loads(data)                                   # (H,W,3) float32 [0..1]
            img_arr = np.asarray(img, dtype=np.float32)
            if img_arr.shape[:2] != (args.obs_h, args.obs_w):
                img_arr = fit_to_hw(img_arr, args.obs_h, args.obs_w)
            obs = jnp.array(img_arr).reshape((1, *obs_shape))          # (1,100,100,3)

            HISTORY.append(obs)
            n = len(HISTORY)
            raw = -1 + np.array(config.frame_offsets, dtype=int)   # -1 is current
            raw = np.maximum(-n, raw)                              # clamp to oldest available

            # convert negatives to positive indices for deque
            idx = [(n + i) if i < 0 else int(i) for i in raw.tolist()]

            obs_stack = [HISTORY[k] for k in idx]                  # pick frames by offsets
            obs_stacked = jnp.concatenate(obs_stack, axis=-1)      # (1,H,W, 3*len(offsets))            

            # ---- match your previously working call pattern ----
            obs_fixed = obs_stacked.copy()

            # 4) forward (explicit kwargs to avoid signature mismatches)
            t_prep = time.perf_counter()
            action_dist = actor_module.apply(
                {"params": actor_params},
                obs_fixed,
                goal_fixed,
                goal_encoded=False,
            )
            if config['discrete'] == True:
                if config['multi_discrete']:
                    act = action_dist.mean()
                else:
                    act = action_dist.mode()
            else:
                act = action_dist.mean()
            t_fwd = time.perf_counter()

            # 5) send action (12 bytes: 3 * float32)
            act_np = np.array(act[0], dtype=np.float32)
            print(act_np)
            conn.sendall(act_np.tobytes())
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

