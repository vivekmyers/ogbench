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
    # Per-pair category tag, one of {"same_traj_easy", "same_traj_hard",
    # "random_goal"}. Auto-classified from dataset terminals + index distance
    # when pairs are given as bare [s, g] in the JSON, or taken directly from
    # the "category" field when the entry is a dict.
    categories: list[str]
    frames_per_episode: int
    goal_xy_mse_threshold: float | None
    num_episodes: int


# Category vocabulary. Keep in sync with client aggregation and server header.
CATEGORY_SAME_EASY = "same_traj_easy"
CATEGORY_SAME_HARD = "same_traj_hard"
CATEGORY_RANDOM = "random_goal"
VALID_CATEGORIES = (CATEGORY_SAME_EASY, CATEGORY_SAME_HARD, CATEGORY_RANDOM)


def classify_pair(
    start: int,
    goal: int,
    *,
    terminals: np.ndarray | None,
    easy_hard_threshold: int,
) -> str:
    """Auto-assign a (start, goal) pair to a category.

    - Same trajectory <=> no ``terminals > 0`` strictly between the two indices
      (we stepped through at most one contiguous episode in the dataset).
    - Within same trajectory, ``|goal - start| <= easy_hard_threshold`` is
      "easy", else "hard". The threshold is in *dataset-index units* (i.e.
      training frames), so at 10 Hz a threshold of 200 ≈ 20 s of expert play.
    - Cross-trajectory <=> "random_goal".

    When ``terminals`` is not available, fall back to distance-only heuristic.
    """
    s = int(min(start, goal))
    g = int(max(start, goal))
    if terminals is not None and g > s:
        # Any non-zero terminal in [s, g-1] means the dataset broke an episode
        # between the two indices — they are not in the same trajectory.
        # (terminal at g itself is fine: it's the last step of g's trajectory.)
        t_slice = terminals[s:g]
        crosses = bool(np.any(np.asarray(t_slice) > 0))
    elif terminals is None:
        crosses = False
    else:
        crosses = False
    if crosses:
        return CATEGORY_RANDOM
    if abs(int(goal) - int(start)) <= int(easy_hard_threshold):
        return CATEGORY_SAME_EASY
    return CATEGORY_SAME_HARD


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
    terminals: np.ndarray | None = None,
    easy_hard_threshold: int = 200,
) -> EvalPlan:
    """Parse eval_config pairs, clip to dataset size, and attach categories.

    Each entry in ``raw["pairs"]`` may be:
      * ``[start, goal]`` — category auto-classified.
      * ``{"start": s, "goal": g}`` or ``{"start": s, "goal": g, "category": c}``
        — explicit start/goal, optional manual category override. ``c`` must be
        one of {"same_traj_easy", "same_traj_hard", "random_goal"}.
    """
    pairs_in = raw["pairs"]
    if not isinstance(pairs_in, list) or len(pairs_in) == 0:
        raise ValueError("eval_config 'pairs' must be a non-empty list")

    pairs: list[tuple[int, int]] = []
    categories: list[str] = []
    # Per-JSON threshold override, CLI arg still wins.
    jthr = raw.get("easy_hard_threshold")
    if jthr is not None:
        easy_hard_threshold = int(jthr)

    for row in pairs_in:
        manual_cat: str | None = None
        if isinstance(row, dict):
            if "start" not in row or "goal" not in row:
                raise ValueError(f"dict pairs entry must have 'start' and 'goal', got {row!r}")
            s_raw, g_raw = row["start"], row["goal"]
            if "category" in row and row["category"] is not None:
                manual_cat = str(row["category"])
                if manual_cat not in VALID_CATEGORIES:
                    raise ValueError(
                        f"category={manual_cat!r} not in {VALID_CATEGORIES}"
                    )
        elif isinstance(row, (list, tuple)) and len(row) == 2:
            s_raw, g_raw = row
        else:
            raise ValueError(f"each pairs entry must be [start, goal] or a dict, got {row!r}")

        s, g = int(s_raw), int(g_raw)
        s = int(np.clip(s, 0, num_dataset_frames - 1))
        g = int(np.clip(g, 0, num_dataset_frames - 1))
        pairs.append((s, g))
        cat = manual_cat if manual_cat is not None else classify_pair(
            s, g, terminals=terminals, easy_hard_threshold=easy_hard_threshold,
        )
        categories.append(cat)

    json_cap = raw.get("max_episodes")
    if json_cap is not None:
        jc = int(json_cap)
        if jc < 1:
            raise ValueError("eval_config max_episodes must be >= 1")
        pairs = pairs[:jc]
        categories = categories[:jc]

    if max_episodes is not None:
        mc = int(max_episodes)
        if mc < 1:
            raise ValueError("max_episodes must be >= 1")
        pairs = pairs[:mc]
        categories = categories[:mc]

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
        categories=categories,
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
    goal_frame_offsets: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (goal_stacked H,W,3*k uint8, gi_obs H,W,3 uint8 last / current-goal frame).

    ``goal_frame_offsets`` matches training: non-positive ints with 0 = goal
    frame, -1 = one step before, etc., sorted oldest-first along channels
    (same convention as ``GCDataset._normalize_frame_offsets``). When
    ``None``, uses consecutive history
    ``(-(K-1), ..., -1, 0)`` with ``K = frame_stack_k`` (legacy behavior).
    """
    if goal_frame_offsets is None:
        offs = tuple(range(-(frame_stack_k - 1), 1))
    else:
        offs = tuple(sorted(int(x) for x in goal_frame_offsets))
        if len(offs) != int(frame_stack_k):
            raise ValueError(
                f"goal_frame_offsets has {len(offs)} entries but frame_stack_k={frame_stack_k}."
            )
        if any(o > 0 for o in offs):
            raise ValueError(f"goal_frame_offsets must be <= 0, got {offs}.")
        if 0 not in offs:
            raise ValueError(f"goal_frame_offsets must include 0, got {offs}.")

    goal_frame_stack: list[np.ndarray] = []
    g_idx = int(goal_idx)
    for o in offs:
        lag = -int(o)
        idx = g_idx - lag
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
    category: str | None = None,
    goal_frame_stack: int | None = None,
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
    if category is not None:
        header["category"] = str(category)
    if goal_frame_stack is not None:
        header["goal_frame_stack"] = int(goal_frame_stack)
    return header


def neutral_action_chunk(action_dim: int, chunk_len: int) -> np.ndarray:
    return np.zeros((chunk_len, action_dim), dtype=np.float32)


def send_action_chunk_wire(conn: socket.socket, act_chunk: np.ndarray) -> None:
    chunk_length = int(act_chunk.shape[0])
    chunk_length_bytes = chunk_length.to_bytes(4, "big")
    action_bytes = np.asarray(act_chunk, dtype=np.float32).tobytes()
    conn.sendall(chunk_length_bytes + action_bytes)
