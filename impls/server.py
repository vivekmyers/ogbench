#!/usr/bin/env python3
import os
import socket
import pickle
import argparse
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

from eval_utils import (
    dataset_image_stack,
    fit_to_hw,
    get_xy_from_dataset,
    get_yaw_deg_from_dataset,
    recvall,
    send_len_pickled,
)
from eval_multi_episode import (
    EvalPlan,
    build_goal_stack_uint8,
    header_dict_from_pair,
    load_eval_config,
    neutral_action_chunk,
    normalize_eval_config,
    send_action_chunk_wire,
)
from agents import agents as AGENT_REGISTRY

# ---------------- XLA memory knobs (same as your setup) ----------------
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

# ---------------- util: import agent family ----------------
def build_agent(agent_str: str):
    agent_str = agent_str.lower().strip()
    if agent_str not in AGENT_REGISTRY:
        raise KeyError(
            f"Unknown agent {agent_str!r}. Available: {sorted(AGENT_REGISTRY)}"
        )
    Agent = AGENT_REGISTRY[agent_str]
    module = importlib.import_module(f"agents.{agent_str}")
    get_config = getattr(module, "get_config")
    return Agent, get_config

def _cfg_pick(config, key, default=None):
    """Read key from ml_collections.ConfigDict or dict-like; treat absent/None as default."""
    try:
        v = config[key]
    except (KeyError, TypeError):
        return default
    return default if v is None else v


def _resolve_eval_dims_from_config(config, args, *, loaded_config_json: bool):
    """Obs geometry and frame_stack must match the checkpoint (from saved training config)."""
    if loaded_config_json:
        obs_h = int(_cfg_pick(config, "obs_h", 64))
        obs_w = int(_cfg_pick(config, "obs_w", 64))
        obs_c = int(_cfg_pick(config, "obs_c", 3))
    elif _cfg_pick(config, "obs_h") is not None:
        obs_h = int(config["obs_h"])
        obs_w = int(_cfg_pick(config, "obs_w", obs_h))
        obs_c = int(_cfg_pick(config, "obs_c", 3))
    else:
        obs_h = int(args.obs_h) if args.obs_h is not None else 64
        obs_w = int(args.obs_w) if args.obs_w is not None else 64
        obs_c = int(args.obs_c) if args.obs_c is not None else 3

    fs = _cfg_pick(config, "frame_stack", None)
    if fs is not None:
        frame_stack_k = int(fs)
    else:
        fo = _cfg_pick(config, "frame_offsets", (0, -1, -2))
        if isinstance(fo, list):
            fo = tuple(fo)
        frame_stack_k = len(fo)
        print(
            f"[WARN] config has no frame_stack; using len(frame_offsets)={frame_stack_k}. "
            "Prefer re-training with frame_stack in config or fix config.json."
        )
    return obs_h, obs_w, obs_c, frame_stack_k


_AGENT_CHOICES = sorted(AGENT_REGISTRY.keys())

def _resolve_frame_offsets(config, frame_stack_k: int) -> tuple[int, ...]:
    """Resolve explicit per-stack offsets (oldest-first, includes 0).

    Offsets are in dataset/policy-step units and must be <= 0.
    Examples:
      - canonical consecutive K=3: (-2, -1, 0)
      - spaced: (0, -20, -40) (will be returned as (-40, -20, 0))
    """
    fo = _cfg_pick(config, "frame_offsets", None)
    if fo is None:
        return tuple(range(-(frame_stack_k - 1), 1))
    if isinstance(fo, list):
        fo = tuple(int(x) for x in fo)
    else:
        fo = tuple(int(x) for x in fo)
    if len(fo) != int(frame_stack_k):
        raise ValueError(
            f"config frame_offsets has {len(fo)} entries but frame_stack={frame_stack_k}."
        )
    if any(x > 0 for x in fo):
        raise ValueError(f"frame_offsets must be <= 0, got {fo}.")
    if 0 not in fo:
        raise ValueError(f"frame_offsets must include 0 (current frame), got {fo}.")
    # Oldest-first
    return tuple(sorted(fo))

def _history_len_for_offsets(frame_offsets: tuple[int, ...]) -> int:
    max_lag = max(0, max((-o for o in frame_offsets if o < 0), default=0))
    return int(max_lag + 1)

# ---------------- main ----------------
def main():
    p = argparse.ArgumentParser()
    # Server-specific arguments
    p.add_argument(
        "--agent",
        default="gcbc",
        choices=_AGENT_CHOICES,
        help=f"Agent module (available: {', '.join(_AGENT_CHOICES)})",
    )
    p.add_argument("--model_path",
                   default="/global/scratch/users/achyuthkv76/tmd_models/run2.pkl")
    p.add_argument("--config_path", type=str, default=None,
                   help="Path to config.json file (saved during training). If provided, agent will be created based on this config.")
    p.add_argument("--dataset_path",
                   default="/nfs/kun2/users/achyuth/carla_test_scripts/goals.npz")
    p.add_argument(
        "--goal_frame_index",
        "--goal_index",
        type=int,
        default=1000,
        dest="goal_frame_index",
        help="Dataset frame index for goal image stack (alias: --goal_index)",
    )
    p.add_argument(
        "--start_frame_index",
        type=int,
        default=0,
        help="Dataset frame index whose (x,y) [and optional yaw] spawn the ego in CARLA (sent to client in header)",
    )
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=5050)
    
    p.add_argument(
        "--obs_h",
        type=int,
        default=None,
        help="Manual override for observation height. If omitted, use config.json (or 64 when no config).",
    )
    p.add_argument(
        "--obs_w",
        type=int,
        default=None,
        help="Manual override for observation width. If omitted, use config.json (or 64 when no config).",
    )
    p.add_argument(
        "--obs_c",
        type=int,
        default=None,
        help="Manual override for channels per frame (usually 3). If omitted, use config.json (or 3 when no config).",
    )
    p.add_argument(
        "--frame_stack",
        type=int,
        default=None,
        help="Manual override for contiguous frame stack depth (must match checkpoint). If omitted, use config.json.",
    )
    p.add_argument(
        "--frame_offsets",
        nargs="*",
        type=int,
        default=None,
        help="Unused when --config_path is set (stacking depth comes from config frame_stack).",
    )
    p.add_argument("--block_size", type=int, default=400, help="Unused for eval I/O when --config_path is set.")
    p.add_argument(
        "--action_chunk_length",
        type=int,
        default=1,
        help="Unused when --config_path is set (use action_chunk_length from config.json).",
    )
    p.add_argument(
        "--n_actions",
        type=int,
        default=None,
        help="Send at most the first N actions from each predicted chunk. "
        "Default: same as action_chunk_length from the merged eval config (config.json, checkpoint "
        "fallback, or agent template). Override for fewer steps per replan (e.g. --n_actions 1).",
    )
    p.add_argument(
        "--brake_deadzone",
        type=float,
        default=0.01,
        help="If brake is in [0, deadzone), force it to 0 before sending to client (post-clip).",
    )
    p.add_argument("--use_discrete", action="store_true", default=False, help="Use discrete actions (multi-discrete mode: discretize throttle/steer/brake into 32 bins each)")
    p.add_argument(
        "--eval_config",
        type=str,
        default=None,
        help="JSON with pairs, frames_per_episode, goal_xy_mse_threshold (see impls/eval_config.json). When set, --start_frame_index/--goal_frame_index are ignored.",
    )
    p.add_argument(
        "--eval_max_episodes",
        type=int,
        default=None,
        help="With --eval_config: run at most this many episodes (first N pairs). Applied after optional max_episodes in the JSON.",
    )
    p.add_argument(
        "--easy_hard_threshold",
        type=int,
        default=200,
        help="Dataset-index gap at or below which a same-trajectory (start,goal) pair "
             "is classified 'same_traj_easy'; above it, 'same_traj_hard'. 10 Hz data "
             "=> 200 ~= 20 s of expert play. Overridden by 'easy_hard_threshold' in "
             "the eval_config JSON if set, and by per-pair 'category' if set.",
    )
    p.add_argument(
        "--accept_multiple",
        action="store_true",
        help="After eval_done (or client disconnect), listen for another connection instead of exiting.",
    )
    args = p.parse_args()

    # ---- Load agent ----
    agent_str = args.agent.lower().strip()
    if agent_str not in AGENT_REGISTRY:
        raise SystemExit(
            f"Unknown --agent {agent_str!r}. Choose one of: {sorted(AGENT_REGISTRY)}"
        )
    Agent = AGENT_REGISTRY[agent_str]
    module = importlib.import_module(f"agents.{agent_str}")
    get_config = getattr(module, "get_config")

    # Load config from file if provided, otherwise try sibling config.json next to the checkpoint
    config = get_config()
    loaded_config_json = False
    config_path = None
    if args.config_path:
        config_path = Path(args.config_path)
        if not config_path.exists():
            raise SystemExit(
                f"--config_path={config_path} does not exist. "
                f"Did you mean {Path(args.model_path).parent / 'config.json'}?"
            )
    else:
        sibling = Path(args.model_path).parent / "config.json"
        if sibling.exists():
            config_path = sibling
            print(f"[INFO] No --config_path given; auto-using {config_path}")

    if config_path is not None:
        loaded_config_json = True
        print(f"Loading config from {config_path}")
        with config_path.open("r") as f:
            config_dict = json.load(f)
        for key, value in config_dict.items():
            if value is None:
                continue  # JSON null must not clobber template (e.g. encoder)
            if isinstance(value, list) and key in [
                "frame_offsets",
                "goal_frame_offsets",
                "actor_hidden_dims",
                "value_hidden_dims",
                "distance_head_hidden_dims",
            ]:
                value = tuple(value)
            if isinstance(config, ml_collections.ConfigDict):
                config[key] = value
            else:
                setattr(config, key, value)
        print(f"Loaded config from file: {len(config_dict)} keys")
    else:
        print("No config file provided, using defaults from agent.get_config()")

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
            # Optional: config fallback from checkpoint if no config.json was loaded
            if not loaded_config_json and "config" in maybe_dict:
                saved_config = maybe_dict["config"]
                if isinstance(saved_config, dict):
                    for key, value in saved_config.items():
                        if value is None:
                            continue
                        if isinstance(value, list) and key in [
                            "frame_offsets",
                            "goal_frame_offsets",
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

    obs_h, obs_w, obs_c, frame_stack_k = _resolve_eval_dims_from_config(
        config, args, loaded_config_json=loaded_config_json
    )
    cli_obs_parts = []
    if args.obs_h is not None:
        obs_h = int(args.obs_h)
        cli_obs_parts.append(f"obs_h={obs_h}")
    if args.obs_w is not None:
        obs_w = int(args.obs_w)
        cli_obs_parts.append(f"obs_w={obs_w}")
    if args.obs_c is not None:
        obs_c = int(args.obs_c)
        cli_obs_parts.append(f"obs_c={obs_c}")
    if cli_obs_parts:
        print(
            f"[EVAL] CLI override: {', '.join(cli_obs_parts)} "
            "(must match checkpoint input shape or forward will fail)"
        )

    if args.frame_stack is not None:
        frame_stack_k = int(args.frame_stack)
        if isinstance(config, ml_collections.ConfigDict):
            config.frame_stack = frame_stack_k
        else:
            config["frame_stack"] = frame_stack_k
        print(
            f"[EVAL] CLI override: frame_stack={frame_stack_k} "
            "(must match the run that produced --model_path; config.json may be from a different experiment)"
        )

    # Resolve explicit offsets for eval-time stacking and history size.
    frame_offsets = _resolve_frame_offsets(config, frame_stack_k)
    history_maxlen = _history_len_for_offsets(frame_offsets)

    # Optional goal-only temporal stack (must match training ``goal_frame_stack``).
    gfs = _cfg_pick(config, "goal_frame_stack", None)
    goal_stack_k = int(gfs) if gfs is not None else int(frame_stack_k)
    gfo_raw = _cfg_pick(config, "goal_frame_offsets", None)
    goal_frame_offsets: tuple[int, ...] | None
    if gfo_raw is None:
        goal_frame_offsets = None
    else:
        if isinstance(gfo_raw, list):
            goal_frame_offsets = tuple(int(x) for x in gfo_raw)
        else:
            goal_frame_offsets = tuple(int(x) for x in gfo_raw)
        if len(goal_frame_offsets) != goal_stack_k:
            raise ValueError(
                f"config goal_frame_offsets has {len(goal_frame_offsets)} entries "
                f"but goal_frame_stack={goal_stack_k}."
            )
    # When not using a separate goal stack, still mirror training by applying
    # the same ``frame_offsets`` as observations (eval used to always use
    # consecutive lags only, which broke custom obs offsets at goal time).
    if gfs is None and goal_frame_offsets is None:
        fo_obs = _cfg_pick(config, "frame_offsets", None)
        if fo_obs is not None:
            if isinstance(fo_obs, list):
                fo_obs = tuple(int(x) for x in fo_obs)
            else:
                fo_obs = tuple(int(x) for x in fo_obs)
            if len(fo_obs) == goal_stack_k:
                goal_frame_offsets = fo_obs

    obs_shape = (obs_h, obs_w, obs_c)
    stacked_obs_shape = (obs_h, obs_w, obs_c * frame_stack_k)
    stacked_goal_shape = (obs_h, obs_w, obs_c * goal_stack_k)
    print(
        f"[EVAL] Resolved from {'config.json' if loaded_config_json else 'config + defaults'}: "
        f"obs_h={obs_h}, obs_w={obs_w}, obs_c={obs_c}, frame_stack={frame_stack_k}, "
        f"goal_frame_stack={goal_stack_k}, "
        f"action_chunk_length={int(_cfg_pick(config, 'action_chunk_length', 1))}"
    )
    if frame_offsets != tuple(range(-(frame_stack_k - 1), 1)):
        print(f"[EVAL] Using explicit frame_offsets={frame_offsets} (history_maxlen={history_maxlen})")
    # Goal tensor differs from observation tensor only when using a separate goal stack.
    if stacked_goal_shape != stacked_obs_shape:
        print(
            f"[EVAL] Goal tensor shape {stacked_goal_shape} "
            f"(goal_frame_offsets={goal_frame_offsets})"
        )

    # Initialize with *stacked* shape (this must match runtime)
    dummy_obs = jnp.zeros((1, *stacked_obs_shape), dtype=jnp.float32)
    dummy_act = jnp.zeros((1, *act_shape), dtype=jnp.float32)  # (1, 3)
    dummy_goal = jnp.zeros((1, *stacked_goal_shape), dtype=jnp.float32)

    # Create agent with config that matches checkpoint (structure will match)
    create_kwargs = dict(
        seed=0,
        ex_observations=dummy_obs,
        ex_actions=dummy_act,
        config=config,
    )
    if agent_str == "gcbc":
        create_kwargs["ex_goals"] = dummy_goal
    agent = Agent.create(**create_kwargs)
    
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
    
    # ---- Load dataset ----
    dataset = np.load(args.dataset_path)
    frames = dataset_image_stack(dataset)
    num_frames = len(frames)

    eval_plan: EvalPlan | None = None
    if args.eval_config:
        raw = load_eval_config(Path(args.eval_config))
        # Pull dataset terminals for auto-classification (same_traj vs random).
        # npz may not have a terminals key for every dataset — pass None and
        # fall back to a distance-only heuristic inside classify_pair.
        terms = None
        if 'terminals' in dataset.files:
            terms = np.asarray(dataset['terminals'])
        eval_plan = normalize_eval_config(
            raw,
            agent_name=args.agent,
            num_dataset_frames=num_frames,
            max_episodes=args.eval_max_episodes,
            terminals=terms,
            easy_hard_threshold=args.easy_hard_threshold,
        )
        # Show per-category counts and a dump of the first few classified
        # pairs so it's obvious from stdout how the test suite was split.
        from collections import Counter as _Counter
        cat_counts = _Counter(eval_plan.categories)
        print(
            f"[EVAL] eval_config: {eval_plan.num_episodes} episodes, "
            f"frames_per_episode={eval_plan.frames_per_episode}, "
            f"goal_xy_mse_threshold={eval_plan.goal_xy_mse_threshold}, "
            f"easy_hard_threshold={args.easy_hard_threshold}"
        )
        print(f"[EVAL] categories: {dict(cat_counts)}")
        for i, ((s, g), c) in enumerate(zip(eval_plan.pairs, eval_plan.categories)):
            print(f"[EVAL]   ep{i:03d} start={s} goal={g} cat={c}")
        s_idx, g_idx = eval_plan.pairs[0]
    else:
        g_idx = int(np.clip(args.goal_frame_index, 0, num_frames - 1))
        s_idx = int(np.clip(args.start_frame_index, 0, num_frames - 1))
        if g_idx != args.goal_frame_index or s_idx != args.start_frame_index:
            print(
                f"[WARN] Clamped goal_frame_index={args.goal_frame_index}->{g_idx}, "
                f"start_frame_index={args.start_frame_index}->{s_idx} (num_frames={num_frames})"
            )

    def build_goal_state(start_i: int, goal_i: int):
        goal_stacked, gi_obs = build_goal_stack_uint8(
            frames,
            goal_i,
            num_frames=num_frames,
            obs_h=obs_h,
            obs_w=obs_w,
            frame_stack_k=goal_stack_k,
            goal_frame_offsets=goal_frame_offsets,
        )
        g_xy = get_xy_from_dataset(dataset, goal_i)
        st_xy = get_xy_from_dataset(dataset, start_i)
        st_yaw = get_yaw_deg_from_dataset(dataset, start_i)
        return goal_stacked, gi_obs, g_xy, st_xy, st_yaw

    goal_stacked, gi_obs, goal_xy, start_xy, start_yaw_deg = build_goal_state(s_idx, g_idx)

    if start_xy is not None:
        print(f"[DATA] start_frame_index={s_idx} -> start_xy={start_xy}, start_yaw_deg={start_yaw_deg}")
    else:
        print(f"[DATA] start_frame_index={s_idx} -> no position key found; client will use random spawn")
    print(f"[DATA] goal_frame_index={g_idx} -> goal_xy={goal_xy}")

    print("Goal loaded")

    # ---- Socket setup ----
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.host, args.port))
    server_sock.listen(1)
    print(f"Listening on {args.host}:{args.port}...")

    # Direct module access (matching your previous working pattern)
    actor_module = agent.network.model_def.modules["actor"]
    actor_params = agent.network.params["modules_actor"]

    cfg_chunk_default = int(_cfg_pick(config, "action_chunk_length", 1))
    action_stack_len = int(_cfg_pick(config, "action_stack_length", 1) or 1)

    print("Runtime config:")
    print(f"  obs_shape={obs_shape}")
    print(f"  stacked_obs_shape={stacked_obs_shape}")
    print(f"  stacked_goal_shape={stacked_goal_shape}")
    print(f"  frame_stack={frame_stack_k}, goal_frame_stack={goal_stack_k}")
    print(f"  action_stack_length={action_stack_len}")
    last_chunk_len = cfg_chunk_default
    n_actions_exec = int(args.n_actions) if args.n_actions is not None else cfg_chunk_default
    if args.n_actions is None:
        print(f"[EVAL] n_actions={n_actions_exec} (default: action_chunk_length from config)")
    else:
        print(f"[EVAL] n_actions={n_actions_exec} (CLI override; action_chunk_length={cfg_chunk_default})")

    def serve_one_client(conn: socket.socket, addr):
        nonlocal last_chunk_len, goal_stacked, gi_obs, goal_xy, start_xy, start_yaw_deg, s_idx, g_idx

        if eval_plan is not None:
            completed_episodes = 0
            num_eps = eval_plan.num_episodes
            pairs = eval_plan.pairs
            s_idx, g_idx = pairs[0]
            goal_stacked, gi_obs, goal_xy, start_xy, start_yaw_deg = build_goal_state(s_idx, g_idx)
            goal_fixed = jnp.array(goal_stacked[None, ...])
            HISTORY = deque(maxlen=history_maxlen)
            action_past = deque(maxlen=int(action_stack_len) if int(action_stack_len) > 1 else None)
            hdr = header_dict_from_pair(
                gi_obs=gi_obs,
                goal_xy=goal_xy,
                start_xy=start_xy,
                start_yaw_deg=start_yaw_deg,
                start_frame_index=s_idx,
                goal_frame_index=g_idx,
                obs_h=obs_h,
                obs_w=obs_w,
                obs_c=obs_c,
                frame_stack=frame_stack_k,
                episode_index=0,
                num_episodes=num_eps,
                frames_per_episode=eval_plan.frames_per_episode,
                goal_xy_mse_threshold=eval_plan.goal_xy_mse_threshold,
                category=eval_plan.categories[0],
                goal_frame_stack=goal_stack_k,
            )
            hdr["frame_offsets"] = list(frame_offsets)
        else:
            num_eps = 1
            completed_episodes = 0
            goal_fixed = jnp.array(goal_stacked[None, ...])
            HISTORY = deque(maxlen=history_maxlen)
            action_past = deque(maxlen=int(action_stack_len) if int(action_stack_len) > 1 else None)
            hdr = header_dict_from_pair(
                gi_obs=gi_obs,
                goal_xy=goal_xy,
                start_xy=start_xy,
                start_yaw_deg=start_yaw_deg,
                start_frame_index=s_idx,
                goal_frame_index=g_idx,
                obs_h=obs_h,
                obs_w=obs_w,
                obs_c=obs_c,
                frame_stack=frame_stack_k,
                episode_index=0,
                num_episodes=1,
                frames_per_episode=None,
                goal_xy_mse_threshold=None,
                goal_frame_stack=goal_stack_k,
            )
            hdr["frame_offsets"] = list(frame_offsets)

        try:
            send_len_pickled(conn, hdr)
            gi_u8 = np.asarray(gi_obs, dtype=np.uint8)
            print(
                f"Sent eval header ep {hdr.get('episode_index')} (goal img {gi_u8.shape}, "
                f"goal_xy={goal_xy}, start_xy={start_xy})"
            )
        except Exception as e:
            print(f"[WARN] failed to send goal header: {e}")

        try:
            while True:
                length_bytes = conn.recv(4)
                if not length_bytes:
                    break
                msg_len = int.from_bytes(length_bytes, "big")
                data = recvall(conn, msg_len, timeout=30.0)

                obj = pickle.loads(data)

                if isinstance(obj, dict) and obj.get("type") == "episode_end":
                    if eval_plan is None:
                        print("[WARN] received episode_end but --eval_config not set; ignoring.")
                        continue
                    completed_episodes += 1
                    print(
                        f"[EVAL] episode_end reason={obj.get('reason')} frames_used={obj.get('frames_used')} "
                        f"completed={completed_episodes}/{num_eps}"
                    )
                    if completed_episodes >= num_eps:
                        send_len_pickled(conn, {"type": "eval_done", "num_episodes": num_eps})
                        print("[EVAL] sent eval_done")
                        break
                    s_idx, g_idx = pairs[completed_episodes]
                    goal_stacked, gi_obs, goal_xy, start_xy, start_yaw_deg = build_goal_state(s_idx, g_idx)
                    goal_fixed = jnp.array(goal_stacked[None, ...])
                    HISTORY.clear()
                    action_past.clear()
                    hdr = header_dict_from_pair(
                        gi_obs=gi_obs,
                        goal_xy=goal_xy,
                        start_xy=start_xy,
                        start_yaw_deg=start_yaw_deg,
                        start_frame_index=s_idx,
                        goal_frame_index=g_idx,
                        obs_h=obs_h,
                        obs_w=obs_w,
                        obs_c=obs_c,
                        frame_stack=frame_stack_k,
                        episode_index=completed_episodes,
                        num_episodes=num_eps,
                        frames_per_episode=eval_plan.frames_per_episode,
                        goal_xy_mse_threshold=eval_plan.goal_xy_mse_threshold,
                        category=eval_plan.categories[completed_episodes],
                        goal_frame_stack=goal_stack_k,
                    )
                    hdr["frame_offsets"] = list(frame_offsets)
                    send_len_pickled(conn, hdr)
                    act_chunk = neutral_action_chunk(action_dim, last_chunk_len)
                    send_action_chunk_wire(conn, act_chunk)
                    print(
                        f"[EVAL] next episode {completed_episodes}: start={s_idx} goal={g_idx}, "
                        f"sent neutral chunk len={last_chunk_len}"
                    )
                    continue

                if isinstance(obj, dict):
                    print(f"[WARN] unexpected dict from client: {list(obj.keys())}")
                    continue

                img_arr = np.asarray(obj)

                if img_arr.shape[:2] != (obs_h, obs_w):
                    img_arr = fit_to_hw(img_arr, obs_h, obs_w)

                if img_arr.dtype != np.uint8:
                    if img_arr.max() <= 1.0:
                        img_arr = np.clip(img_arr * 255.0, 0, 255).astype(np.uint8)
                    else:
                        img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)

                # Detect a client that already stacked frames at policy-step
                # cadence (channel dim = obs_c * frame_stack_k). For such clients
                # we bypass the server's per-request HISTORY (which would stack
                # at chunk-cadence and break temporal alignment with training).
                # Old single-frame clients (channel dim == obs_c) still hit the
                # legacy HISTORY path. When frame_stack_k == 1 the two are
                # equivalent and we just take the fast path.
                last_axis = int(img_arr.shape[-1])
                if last_axis == obs_c * frame_stack_k:
                    obs_fixed = jnp.array(img_arr).reshape((1, *stacked_obs_shape))
                    HISTORY.append(obs_fixed)  # only kept for `n` debug counter
                    n = len(HISTORY)
                elif last_axis == obs_c:
                    obs = jnp.array(img_arr).reshape((1, *obs_shape))
                    HISTORY.append(obs)
                    # Stack using explicit offsets (oldest-first). HISTORY stores
                    # single frames at policy-step cadence.
                    n = len(HISTORY)
                    obs_stack = []
                    for off in frame_offsets:
                        idx = n - 1 + int(off)
                        idx = max(0, idx)
                        obs_stack.append(HISTORY[idx])
                    obs_fixed = jnp.concatenate(obs_stack, axis=-1).copy()
                else:
                    raise ValueError(
                        f"Unexpected obs channel dim {last_axis}; expected {obs_c} (single frame) "
                        f"or {obs_c * frame_stack_k} (pre-stacked from client)."
                    )

                obs_mean = float(np.array(obs_fixed).mean())
                print(f"[DEBUG step {n}] obs mean={obs_mean:.6f}")

                if n == 1:
                    print("\n[DEBUG] Input shapes and ranges (matching training format):")
                    print(
                        f"  obs_fixed: shape={obs_fixed.shape}, dtype={obs_fixed.dtype}, "
                        f"range=[{obs_fixed.min():.3f}, {obs_fixed.max():.3f}]"
                    )
                    print(
                        f"  goal_fixed: shape={goal_fixed.shape}, dtype={goal_fixed.dtype}, "
                        f"range=[{goal_fixed.min():.3f}, {goal_fixed.max():.3f}]"
                    )

                variables = {"params": actor_params}
                apply_kwargs = {
                    "observations": obs_fixed,
                    "goals": goal_fixed,
                    "goal_encoded": False,
                    "temperature": 1.0,
                }
                if action_stack_len > 1:
                    # Match training-time padding semantics: when the stack is shorter
                    # than L at the start of an episode, repeat the earliest available
                    # action; if nothing exists yet, fall back to neutral (all zeros).
                    _neutral = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    _h = list(action_past)
                    _L = int(action_stack_len)
                    if len(_h) < _L:
                        _pad = _h[0] if len(_h) > 0 else _neutral
                        _rows = [_pad] * (_L - len(_h)) + _h
                    else:
                        _rows = _h[-_L:]
                    apply_kwargs["action_stack"] = jnp.array(
                        np.stack(_rows, axis=0)[None, ...], dtype=jnp.float32
                    )
                    if n <= 3:
                        _as_np = np.stack(_rows, axis=0)
                        print(
                            f"[DEBUG step {n}] action_stack shape={_as_np.shape} "
                            f"mean={_as_np.mean():.3f} std={_as_np.std():.3f} "
                            f"last={_as_np[-1].round(3).tolist()}"
                        )

                action_dist = actor_module.apply(variables, **apply_kwargs)
                act = action_dist.mean()

                act_np = np.array(act[0], dtype=np.float32).ravel()
                flat = int(act_np.size)
                if flat % action_dim != 0:
                    raise ValueError(
                        f"Actor output length {flat} is not divisible by action_dim={action_dim}."
                    )
                chunk_len_effective = flat // action_dim
                last_chunk_len = chunk_len_effective
                cfg_chunk = int(_cfg_pick(config, "action_chunk_length", 1))
                if cfg_chunk != chunk_len_effective and n <= 3:
                    print(
                        f"[WARN] config action_chunk_length={cfg_chunk} but actor output implies "
                        f"chunk_len={chunk_len_effective}; using output shape."
                    )
                act_chunk = act_np.reshape(chunk_len_effective, action_dim)
                chunk_length = act_chunk.shape[0]

                if chunk_length > n_actions_exec:
                    original_chunk_length = chunk_length
                    act_chunk = act_chunk[:n_actions_exec]
                    chunk_length = n_actions_exec
                    print(
                        f"[INFO] Using only first {chunk_length} actions from chunk "
                        f"(model predicted {original_chunk_length} actions)"
                    )

                act_chunk[:, 0] = np.clip(act_chunk[:, 0], 0.0, 1.0)
                act_chunk[:, 1] = np.clip(act_chunk[:, 1], -1.0, 1.0)
                act_chunk[:, 2] = np.clip(act_chunk[:, 2], 0.0, 1.0)
                bd = float(getattr(args, "brake_deadzone", 0.0) or 0.0)
                if bd > 0.0:
                    act_chunk[:, 2] = np.where(act_chunk[:, 2] < bd, 0.0, act_chunk[:, 2])

                if action_stack_len > 1:
                    for _i in range(int(chunk_length)):
                        action_past.append(np.asarray(act_chunk[_i], dtype=np.float32).copy())

                send_action_chunk_wire(conn, act_chunk)

        except KeyboardInterrupt:
            print("Interrupted by User")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    try:
        while True:
            conn, addr = server_sock.accept()
            print(f"Connection established with client: {addr}")
            serve_one_client(conn, addr)
            if not args.accept_multiple:
                break
            print("[EVAL] Waiting for next client (--accept_multiple)…")
    finally:
        try:
            server_sock.close()
        except Exception:
            pass
        print("Server shutdown cleanly")

if __name__ == "__main__":
    main()
