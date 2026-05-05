#!/usr/bin/env python3
"""Save one top-down trajectory map (PNG) per episode in a training .npz.

Uses the same matplotlib Agg rendering as eval ``client.py`` (trajectory + start/end markers;
no CARLA road map). Run from anywhere:

  python plot_dataset_trajectories.py --dataset /path/to/data.npz --out-dir plots/my_run

Requires array ``position`` with shape (T, >=2) where columns 0,1 are world x, y.
Episode boundaries come from ``terminals`` (or ``dones`` / ``episode_ends`` / ``is_terminal``),
same heuristics as ``train.py`` when markers are missing.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_IMPLS = Path(__file__).resolve().parent
if str(_IMPLS) not in sys.path:
    sys.path.insert(0, str(_IMPLS))

from utils.trajectory_plot import (  # noqa: E402
    render_trajectory_map_rgb,
    save_trajectory_map_png,
    trajectory_slices,
)


def _maybe_get_terminals(data: np.lib.npyio.NpzFile | dict) -> np.ndarray | None:
    if "terminals" in data:
        t = np.asarray(data["terminals"], dtype=bool).reshape(-1)
        if t.size > 0 and t.any():
            return t
    for key in ("dones", "episode_ends", "is_terminal"):
        if key in data:
            t = np.asarray(data[key], dtype=bool).reshape(-1)
            if t.size > 0 and t.any():
                return t
    return None


def _build_terminals(data: np.lib.npyio.NpzFile, total_frames: int) -> np.ndarray:
    t = _maybe_get_terminals(data)
    if t is not None:
        out = t.astype(bool).copy()
    else:
        out = np.zeros(total_frames, dtype=bool)
        step = 1000
        out[np.arange(step - 1, total_frames, step, dtype=int)] = True
    out[-1] = True
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot dataset trajectories (x,y) to PNG files.")
    ap.add_argument("--dataset", type=str, required=True, help="Path to .npz (must contain position)")
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: impls/plots/dataset_trajectories_<timestamp>)",
    )
    ap.add_argument("--max", type=int, default=None, help="Max number of trajectory PNGs to write")
    ap.add_argument("--mmap", action="store_true", help="Memory-map the .npz (large files)")
    args = ap.parse_args()

    path = Path(args.dataset).expanduser().resolve()
    if not path.is_file():
        sys.exit(f"Not a file: {path}")

    mmap_mode = "r" if args.mmap else None
    data = np.load(path, mmap_mode=mmap_mode)

    if "position" not in data:
        sys.exit(".npz must contain a 'position' array (T, >=2) with x,y in the first two columns.")

    pos = np.asarray(data["position"], dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] < 2:
        sys.exit(f"position must be 2D with at least 2 columns; got shape {pos.shape}")

    T = int(pos.shape[0])
    terminals = _build_terminals(data, T)
    if terminals.shape[0] != T:
        sys.exit(f"terminals length {terminals.shape[0]} != position rows {T}")

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (_IMPLS / "plots" / f"dataset_trajectories_{ts}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for ti, (start, end) in enumerate(trajectory_slices(terminals)):
        if args.max is not None and n_written >= args.max:
            break
        seg = pos[start:end]
        if seg.shape[0] == 0:
            continue
        xs = seg[:, 0].tolist()
        ys = seg[:, 1].tolist()
        start_xy = (float(xs[0]), float(ys[0]))
        goal_xy = (float(xs[-1]), float(ys[-1]))
        rgb = render_trajectory_map_rgb(xs, ys, goal_xy=goal_xy, start_xy=start_xy, map_xy=None)
        if rgb is None:
            continue
        save_trajectory_map_png(out_dir / f"trajectory_{ti:04d}.png", rgb)
        n_written += 1

    data.close()
    print(f"Wrote {n_written} PNGs under {out_dir}")


if __name__ == "__main__":
    main()
