"""Matplotlib trajectory maps (Agg backend) shared by client eval and offline dataset dumps."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def _finite_xy_pairs(xs: Sequence[float], ys: Sequence[float]) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)]


def render_trajectory_map_rgb(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    goal_xy=None,
    start_xy=None,
    map_xy: np.ndarray | Sequence[tuple[float, float]] | None = None,
    figsize: tuple[float, float] = (6, 6),
    dpi: int = 120,
) -> np.ndarray | None:
    """Top-down trajectory plot; returns RGB uint8 (H, W, 3) or None if no trajectory points.

    Same visual language as ``client.wandb_trajectory_map_image`` (map optional).
    ``map_xy`` may be an (N, 2) float array or an iterable of (x, y) pairs.
    """
    traj_pts = _finite_xy_pairs(xs, ys)
    if not traj_pts:
        return None

    fig = Figure(figsize=figsize, dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    if map_xy is not None:
        mp = np.asarray(map_xy, dtype=np.float64).reshape(-1, 2)
        ok = np.isfinite(mp).all(axis=1)
        mp = mp[ok]
        if mp.shape[0] > 0:
            ax.scatter(mp[:, 0], mp[:, 1], s=2, c="#cccccc", alpha=0.3, label="map")

    tx, ty = zip(*traj_pts)
    ax.plot(tx, ty, c="#4e79a7", linewidth=0.5, alpha=0.8, label="traj", zorder=2)
    if len(traj_pts) > 1:
        arrow_interval = max(1, len(traj_pts) // 10)
        for j in range(0, len(traj_pts) - 1, arrow_interval):
            x1, y1 = traj_pts[j]
            x2, y2 = traj_pts[j + 1]
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                ax.annotate(
                    "",
                    xy=(x2, y2),
                    xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="->",
                        lw=2.5,
                        color="#4e79a7",
                        alpha=0.9,
                        mutation_scale=25,
                    ),
                    zorder=3,
                )

    if goal_xy is not None and np.all(np.isfinite(goal_xy)):
        ax.scatter(
            [float(goal_xy[0])],
            [float(goal_xy[1])],
            marker="*",
            s=220,
            c="#f28e2b",
            label="goal",
            zorder=4,
        )
    if start_xy is not None and np.all(np.isfinite(start_xy)):
        ax.scatter(
            [float(start_xy[0])],
            [float(start_xy[1])],
            s=100,
            c="#2ca02c",
            marker="s",
            label="dataset_start_xy",
            zorder=5,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    fig.tight_layout()
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    wh = fig.canvas.get_width_height()[::-1]
    return buf.reshape(wh + (3,))


def save_trajectory_map_png(path: Path | str, rgb: np.ndarray) -> None:
    """Write RGB image to PNG using OpenCV (same as ``client.py``)."""
    import cv2

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(p), bgr):
        raise OSError(f"cv2.imwrite failed for {p}")


def trajectory_slices(terminals: np.ndarray):
    """Yield (start, end) indices per trajectory segment; ``end`` is exclusive (Python slice)."""
    terminals = np.asarray(terminals, dtype=bool).reshape(-1)
    T = int(terminals.shape[0])
    terminal_locs = np.nonzero(terminals)[0]
    if terminal_locs.size == 0:
        yield 0, T
        return
    traj_starts = np.concatenate([[0], terminal_locs[:-1] + 1])
    traj_ends = terminal_locs + 1
    for start, end in zip(traj_starts.astype(int), traj_ends.astype(int)):
        yield int(start), int(end)
