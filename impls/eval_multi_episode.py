"""Multi-episode eval: load eval_config.json, build goals/headers, neutral action chunks."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eval_utils import fit_to_hw


@dataclass(frozen=True)
class EvalPlan:
    pairs: list[tuple[int, int]]
    frames_per_episode: int
    goal_xy_mse_threshold: float | None
    num_episodes: int


def load_eval_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"eval_config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("eval_config must be a JSON object")
    if "pairs" not in raw:
        raise ValueError("eval_config must contain 'pairs'")
    return raw


def normalize_eval_config(
    raw: dict[str, Any],
    *,
    agent_name: str,
    num_dataset_frames: int,
    max_episodes: int | None = None,
) -> EvalPlan:
    pairs_in = raw["pairs"]
    if not isinstance(pairs_in, list) or len(pairs_in) == 0:
        raise ValueError("eval_config 'pairs' must be a non-empty list")
    pairs: list[tuple[int, int]] = []
    for row in pairs_in:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError(f"each pairs entry must be [start, goal], got {row!r}")
        s, g = int(row[0]), int(row[1])
        s = int(np.clip(s, 0, num_dataset_frames - 1))
        g = int(np.clip(g, 0, num_dataset_frames - 1))
        pairs.append((s, g))

    json_cap = raw.get("max_episodes")
    if json_cap is not None:
        jc = int(json_cap)
        if jc < 1:
            raise ValueError("eval_config max_episodes must be >= 1")
        pairs = pairs[:jc]

    if max_episodes is not None:
        mc = int(max_episodes)
        if mc < 1:
            raise ValueError("max_episodes must be >= 1")
        pairs = pairs[:mc]

    if len(pairs) == 0:
        raise ValueError("after max_episodes caps, pairs is empty")

    fpe = raw.get("frames_per_episode", 1500)
    frames_per_episode = int(fpe)

    thr = raw.get("goal_xy_mse_threshold")
    goal_xy_mse_threshold = None if thr is None else float(thr)

    algo = raw.get("algorithm")
    if algo is not None and str(algo).lower() != str(agent_name).lower():
        raise ValueError(
            f"eval_config algorithm={algo!r} does not match --agent={agent_name!r}"
        )

    return EvalPlan(
        pairs=pairs,
        frames_per_episode=frames_per_episode,
        goal_xy_mse_threshold=goal_xy_mse_threshold,
        num_episodes=len(pairs),
    )


def build_goal_stack_uint8(
    frames: np.ndarray,
    goal_idx: int,
    *,
    num_frames: int,
    obs_h: int,
    obs_w: int,
    frame_stack_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (goal_stacked H,W,3*k uint8, gi_obs H,W,3 uint8 last frame)."""
    goal_frame_stack: list[np.ndarray] = []
    g_idx = int(goal_idx)
    for j in range(frame_stack_k):
        idx = g_idx - (frame_stack_k - 1 - j)
        idx = max(0, min(idx, num_frames - 1))
        gi_raw = np.asarray(frames[idx])
        gi_frame = fit_to_hw(gi_raw, obs_h, obs_w)
        if gi_frame.dtype != np.uint8:
            if gi_frame.max() <= 1.0:
                gi_frame = (gi_frame * 255.0).astype(np.uint8)
            else:
                gi_frame = np.clip(gi_frame, 0, 255).astype(np.uint8)
        goal_frame_stack.append(gi_frame)

    goal_stacked = np.concatenate(goal_frame_stack, axis=-1)
    gi_obs = goal_frame_stack[-1]
    return goal_stacked, gi_obs


def header_dict_from_pair(
    *,
    gi_obs: np.ndarray,
    goal_xy: tuple[float, float] | None,
    start_xy: tuple[float, float] | None,
    start_yaw_deg: float | None,
    start_frame_index: int,
    goal_frame_index: int,
    obs_h: int,
    obs_w: int,
    obs_c: int,
    frame_stack: int,
    episode_index: int,
    num_episodes: int,
    frames_per_episode: int | None,
    goal_xy_mse_threshold: float | None,
) -> dict[str, Any]:
    gi_u8 = np.asarray(gi_obs, dtype=np.uint8)
    header: dict[str, Any] = {
        "goal_img": {
            "shape": list(gi_u8.shape),
            "dtype": "uint8",
            "data": gi_u8.reshape(-1).tolist(),
        },
        "goal_xy": goal_xy,
        "start_xy": start_xy,
        "start_yaw_deg": start_yaw_deg,
        "start_frame_index": int(start_frame_index),
        "goal_frame_index": int(goal_frame_index),
        "obs_h": int(obs_h),
        "obs_w": int(obs_w),
        "obs_c": int(obs_c),
        "frame_stack": int(frame_stack),
        "episode_index": int(episode_index),
        "num_episodes": int(num_episodes),
    }
    if frames_per_episode is not None:
        header["frames_per_episode"] = int(frames_per_episode)
    if goal_xy_mse_threshold is not None:
        header["goal_xy_mse_threshold"] = float(goal_xy_mse_threshold)
    return header


def neutral_action_chunk(action_dim: int, chunk_len: int) -> np.ndarray:
    return np.zeros((chunk_len, action_dim), dtype=np.float32)


def send_action_chunk_wire(conn: socket.socket, act_chunk: np.ndarray) -> None:
    chunk_length = int(act_chunk.shape[0])
    chunk_length_bytes = chunk_length.to_bytes(4, "big")
    action_bytes = np.asarray(act_chunk, dtype=np.float32).tobytes()
    conn.sendall(chunk_length_bytes + action_bytes)
