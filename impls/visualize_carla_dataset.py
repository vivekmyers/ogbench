#!/usr/bin/env python3
"""
Lightweight CARLA dataset visualizer + W&B streamer.

Usage (from repo root):

  python -m impls.visualize_carla_dataset \\
      --dataset_path /path/to/dataset.npz \\
      --project carla-dataset-viz

This script:
  - Loads a .npz dataset with at least: 'observations', 'actions', 'terminals'
  - Logs basic stats (shapes, mins/maxes) to stdout and W&B
  - Logs a sampled video of observations to W&B
  - Logs per-dimension action histograms to W&B

No gym/gymnasium/ogbench dependencies; just numpy + wandb.
"""

import argparse
from pathlib import Path

import numpy as np
import wandb


def observations_to_video(observations: np.ndarray, max_frames: int = 1000, fps: int = 10) -> wandb.Video:
    """Convert observations to a W&B Video.

    Expects observations as (T, H, W, C) or (T, H, W).
    """
    # Limit number of frames
    T = observations.shape[0]
    if T > max_frames:
        idxs = np.linspace(0, T - 1, max_frames, dtype=int)
        observations = observations[idxs]

    # Normalize / cast to uint8 [0, 255]
    obs = observations
    if obs.dtype in (np.float32, np.float64):
        if obs.max() <= 1.0:
            obs = (obs * 255.0).clip(0, 255).astype(np.uint8)
        else:
            obs = obs.clip(0, 255).astype(np.uint8)
    elif obs.dtype != np.uint8:
        obs = obs.astype(np.uint8)

    # Handle grayscale
    if obs.ndim == 3:
        # (T, H, W) -> (T, H, W, 3)
        obs = np.stack([obs] * 3, axis=-1)
    elif obs.shape[-1] == 1:
        # (T, H, W, 1) -> (T, H, W, 3)
        obs = np.repeat(obs, 3, axis=-1)

    # W&B expects (T, C, H, W)
    obs = np.transpose(obs, (0, 3, 1, 2))
    return wandb.Video(obs, fps=fps, format="mp4")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize CARLA .npz dataset and stream to W&B.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to CARLA .npz dataset")
    parser.add_argument("--project", type=str, default="carla-dataset-viz", help="W&B project name")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity (optional)")
    parser.add_argument("--run_name", type=str, default=None, help="Optional explicit run name")
    parser.add_argument("--max_frames", type=int, default=1000, help="Max frames in the logged video")
    parser.add_argument("--fps", type=int, default=10, help="FPS for the logged video")
    parser.add_argument(
        "--mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode (default: online)",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Loading dataset from {dataset_path} ...")
    data_np = np.load(dataset_path)

    required_keys = ["observations", "actions", "terminals"]
    for k in required_keys:
        if k not in data_np:
            raise KeyError(f"Dataset missing required key '{k}'")

    obs = np.asarray(data_np["observations"])
    actions = np.asarray(data_np["actions"])
    terminals = np.asarray(data_np["terminals"])

    print("Dataset summary:")
    print(f"  observations: shape={obs.shape}, dtype={obs.dtype}")
    print(f"  actions:      shape={actions.shape}, dtype={actions.dtype}")
    print(f"  terminals:    shape={terminals.shape}, dtype={terminals.dtype}, sum={terminals.sum()}")

    # Basic sanity checks
    assert obs.shape[0] == actions.shape[0] == terminals.shape[0], "Time dimension mismatch"

    # Initialize W&B
    run_name = args.run_name or f"{dataset_path.stem}_viz"
    wandb.init(
        project=args.project,
        entity=args.entity,
        name=run_name,
        config={
            "dataset_path": str(dataset_path),
            "num_frames": int(obs.shape[0]),
            "obs_shape": tuple(obs.shape[1:]),
            "obs_dtype": str(obs.dtype),
            "actions_shape": tuple(actions.shape[1:]),
        },
        mode=args.mode,
    )

    log_dict = {}

    # Video of observations
    try:
        video = observations_to_video(obs, max_frames=args.max_frames, fps=args.fps)
        log_dict["videos/observations"] = video
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: Failed to create video: {e}")

    # Action statistics + histograms
    log_dict["actions/mean"] = float(actions.mean())
    log_dict["actions/std"] = float(actions.std())
    log_dict["actions/min"] = float(actions.min())
    log_dict["actions/max"] = float(actions.max())

    if actions.ndim == 2 and actions.shape[1] >= 1:
        names = ["steer", "throttle", "brake"]
        dim = actions.shape[1]
        for i in range(dim):
            name = names[i] if i < len(names) else f"component_{i}"
            log_dict[f"actions/{name}_hist"] = wandb.Histogram(actions[:, i])

    # Terminals
    log_dict["dataset/num_terminals"] = int(terminals.sum())

    wandb.log(log_dict)
    print(f"Logged dataset visualization to W&B run: {wandb.run.name if wandb.run else 'N/A'}")
    print(f"URL: {wandb.run.url if wandb.run else 'N/A'}")
    wandb.finish()


if __name__ == "__main__":
    main()




