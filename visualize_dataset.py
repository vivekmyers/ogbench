#!/usr/bin/env python3
"""
Script to visualize dataset observations as videos in wandb.

Usage:
    python visualize_dataset.py --dataset_path path/to/dataset.npz
    python visualize_dataset.py --dataset_name antmaze-large-navigate-v0
"""
import argparse
import os
import sys

import numpy as np
import wandb

# Add ogbench to path if not installed
try:
    import ogbench
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ogbench


def is_pixel_observation(obs_shape):
    """
    Check if observations are pixel-based.
    
    Pixel observations typically have:
    - 3 or 4 dimensions (H, W, C) or (N, H, W, C)
    - Height and width > 1
    - Channels in [1, 4] (grayscale or RGB/RGBA)
    """
    if len(obs_shape) == 3:
        h, w, c = obs_shape
        return h > 1 and w > 1 and 1 <= c <= 4
    elif len(obs_shape) == 4:
        n, h, w, c = obs_shape
        return h > 1 and w > 1 and 1 <= c <= 4
    return False


def observations_to_video(observations, max_frames=1000, fps=30):
    """
    Convert observations to wandb video format.
    
    Args:
        observations: numpy array of shape (N, H, W, C) or (N, H, W)
        max_frames: Maximum number of frames to include in video
        fps: Frames per second for the video
    
    Returns:
        wandb.Video object
    """
    # Limit the number of frames
    if len(observations) > max_frames:
        print(f"Limiting video to {max_frames} frames (dataset has {len(observations)})")
        # Sample uniformly across the dataset
        indices = np.linspace(0, len(observations) - 1, max_frames, dtype=int)
        observations = observations[indices]
    
    # Ensure observations are in uint8 format [0, 255]
    if observations.dtype == np.float32 or observations.dtype == np.float64:
        if observations.max() <= 1.0:
            observations = (observations * 255).astype(np.uint8)
        else:
            observations = observations.astype(np.uint8)
    elif observations.dtype != np.uint8:
        observations = observations.astype(np.uint8)
    
    # Handle grayscale by converting to RGB
    if len(observations.shape) == 3:
        # (N, H, W) -> (N, H, W, 3)
        observations = np.stack([observations] * 3, axis=-1)
    elif observations.shape[-1] == 1:
        # (N, H, W, 1) -> (N, H, W, 3)
        observations = np.repeat(observations, 3, axis=-1)
    
    # Convert from (N, H, W, C) to (N, C, H, W) for wandb
    # wandb expects channels-first format for better video rendering
    print(f"Before transpose - shape: {observations.shape}, dtype: {observations.dtype}")
    observations = np.transpose(observations, (0, 3, 1, 2))
    print(f"After transpose - shape: {observations.shape}, dtype: {observations.dtype}")
    
    return wandb.Video(observations, fps=fps, format='mp4')


def visualize_trajectories(dataset, max_trajs=10, max_frames_per_traj=200, fps=30):
    """
    Visualize multiple trajectories from the dataset.
    
    Args:
        dataset: Dictionary with 'observations' and 'terminals'
        max_trajs: Maximum number of trajectories to visualize
        max_frames_per_traj: Maximum frames per trajectory
        fps: Frames per second
    
    Returns:
        List of wandb.Video objects
    """
    observations = dataset['observations']
    terminals = dataset['terminals']
    
    # Find trajectory boundaries
    terminal_indices = np.where(terminals > 0)[0]
    traj_starts = [0] + (terminal_indices[:-1] + 1).tolist()
    traj_ends = terminal_indices.tolist()
    
    print(f"Found {len(traj_starts)} trajectories in the dataset")
    
    videos = []
    for i in range(min(max_trajs, len(traj_starts))):
        start = traj_starts[i]
        end = traj_ends[i] + 1
        traj_obs = observations[start:end]
        
        print(f"Trajectory {i+1}: frames {start} to {end} (length: {len(traj_obs)})")
        
        # Limit frames per trajectory
        if len(traj_obs) > max_frames_per_traj:
            indices = np.linspace(0, len(traj_obs) - 1, max_frames_per_traj, dtype=int)
            traj_obs = traj_obs[indices]
        
        video = observations_to_video(traj_obs, max_frames=len(traj_obs), fps=fps)
        videos.append(video)
    
    return videos


def main():
    parser = argparse.ArgumentParser(
        description='Visualize dataset observations as videos in wandb'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        help='Path to the dataset .npz file'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        help='Name of the OGBench dataset (e.g., visual-cube-lift-v0)'
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='~/.ogbench/data',
        help='Directory where datasets are stored (default: ~/.ogbench/data)'
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default='ogbench-dataset-viz',
        help='Wandb project name'
    )
    parser.add_argument(
        '--wandb_entity',
        type=str,
        default=None,
        help='Wandb entity (username or team)'
    )
    parser.add_argument(
        '--max_frames',
        type=int,
        default=1000,
        help='Maximum number of frames to show in full dataset video'
    )
    parser.add_argument(
        '--max_trajs',
        type=int,
        default=10,
        help='Maximum number of trajectories to visualize separately'
    )
    parser.add_argument(
        '--max_frames_per_traj',
        type=int,
        default=200,
        help='Maximum frames per trajectory'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second for videos'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='online',
        choices=['online', 'offline', 'disabled'],
        help='Wandb mode'
    )
    
    args = parser.parse_args()
    
    # Load dataset
    if args.dataset_path:
        print(f"Loading dataset from: {args.dataset_path}")
        dataset_path = args.dataset_path
        dataset_name = os.path.basename(dataset_path).replace('.npz', '')
        dataset = ogbench.load_dataset(dataset_path)
    elif args.dataset_name:
        print(f"Loading dataset: {args.dataset_name}")
        dataset_name = args.dataset_name
        dataset_dir = os.path.expanduser(args.dataset_dir)
        dataset_path = os.path.join(dataset_dir, f"{dataset_name}.npz")
        
        # Download if needed
        if not os.path.exists(dataset_path):
            print(f"Dataset not found locally. Downloading...")
            ogbench.download_datasets([dataset_name], dataset_dir=dataset_dir)
        
        dataset = ogbench.load_dataset(dataset_path)
    else:
        print("Error: Either --dataset_path or --dataset_name must be provided")
        parser.print_help()
        sys.exit(1)
    
    print(f"\nDataset loaded successfully!")
    print(f"  Observations shape: {dataset['observations'].shape}")
    print(f"  Observations dtype: {dataset['observations'].dtype}")
    print(f"  Actions shape: {dataset['actions'].shape}")
    print(f"  Terminals shape: {dataset['terminals'].shape}")
    
    # Check if observations are pixel-based
    obs_shape = dataset['observations'].shape[1:]
    is_pixels = is_pixel_observation(obs_shape)
    
    print(f"\nObservation type: {'Pixels' if is_pixels else 'States'}")
    
    if not is_pixels:
        print("\nWarning: Observations don't appear to be pixel-based.")
        print("This script is designed for visualizing image observations.")
        print("State-based observations cannot be visualized as videos.")
        
        response = input("\nDo you want to continue and log statistics? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Initialize wandb
    print(f"\nInitializing wandb...")
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"{dataset_name}_visualization",
        config={
            'dataset_name': dataset_name,
            'dataset_path': dataset_path,
            'observations_shape': dataset['observations'].shape,
            'observations_dtype': str(dataset['observations'].dtype),
            'num_transitions': len(dataset['observations']),
            'num_terminals': int(np.sum(dataset['terminals'])),
            'is_pixels': is_pixels,
        },
        mode=args.mode,
    )
    
    # Log dataset statistics
    log_dict = {
        'dataset/num_transitions': len(dataset['observations']),
        'dataset/num_trajectories': int(np.sum(dataset['terminals'])),
        'dataset/obs_shape_h': obs_shape[0] if len(obs_shape) >= 2 else 0,
        'dataset/obs_shape_w': obs_shape[1] if len(obs_shape) >= 2 else 0,
        'dataset/obs_shape_c': obs_shape[2] if len(obs_shape) >= 3 else 0,
    }
    
    if is_pixels:
        print("\nCreating videos...")
        
        # Create full dataset video (sampled)
        print(f"\n1. Creating sampled video of full dataset...")
        full_video = observations_to_video(
            dataset['observations'],
            max_frames=args.max_frames,
            fps=args.fps
        )
        log_dict['videos/full_dataset_sampled'] = full_video
        
        # Create trajectory videos
        print(f"\n2. Creating individual trajectory videos...")
        traj_videos = visualize_trajectories(
            dataset,
            max_trajs=args.max_trajs,
            max_frames_per_traj=args.max_frames_per_traj,
            fps=args.fps
        )
        
        for i, video in enumerate(traj_videos):
            log_dict[f'videos/trajectory_{i+1:03d}'] = video
        
        print(f"\nCreated {len(traj_videos)} trajectory videos")
    else:
        print("\nSkipping video creation for non-pixel observations")
        
        # Log some statistics about state observations
        obs_data = dataset['observations']
        log_dict.update({
            'dataset/obs_mean': np.mean(obs_data),
            'dataset/obs_std': np.std(obs_data),
            'dataset/obs_min': np.min(obs_data),
            'dataset/obs_max': np.max(obs_data),
        })
    
    # Log everything to wandb
    print("\nLogging to wandb...")
    wandb.log(log_dict)
    
    print(f"\n✓ Successfully logged dataset visualization to wandb!")
    print(f"  Project: {args.wandb_project}")
    print(f"  Run: {dataset_name}_visualization")
    print(f"  URL: {wandb.run.url if wandb.run else 'N/A'}")
    
    wandb.finish()


if __name__ == '__main__':
    main()

