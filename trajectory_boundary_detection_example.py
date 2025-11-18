#!/usr/bin/env python3
"""
Example demonstrating trajectory boundary detection in frame stacking.

This example shows how the improved frame stacking function detects trajectory
boundaries by comparing consecutive frames and prevents mixing frames from
different trajectories.
"""
import numpy as np
import sys
import os

# Add impls to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'impls'))

from train import create_spaced_frame_stack, detect_trajectory_boundaries


def create_synthetic_trajectories():
    """
    Create synthetic video data with 3 distinct trajectories.
    Each trajectory has a different background color.
    """
    H, W, C = 64, 64, 3
    traj_length = 20
    
    # Trajectory 1: Red-ish background, moving white square
    traj1 = []
    for t in range(traj_length):
        frame = np.ones((H, W, C), dtype=np.float32) * 0.3
        frame[:, :, 0] = 0.5  # Red tint
        # Moving white square
        x = int((t / traj_length) * (W - 10))
        frame[20:30, x:x+10, :] = 1.0
        traj1.append(frame)
    
    # Trajectory 2: Green-ish background, moving white circle
    traj2 = []
    for t in range(traj_length):
        frame = np.ones((H, W, C), dtype=np.float32) * 0.3
        frame[:, :, 1] = 0.5  # Green tint
        # Moving white square
        x = int((t / traj_length) * (W - 10))
        frame[35:45, x:x+10, :] = 1.0
        traj2.append(frame)
    
    # Trajectory 3: Blue-ish background, moving white square
    traj3 = []
    for t in range(traj_length):
        frame = np.ones((H, W, C), dtype=np.float32) * 0.3
        frame[:, :, 2] = 0.5  # Blue tint
        # Moving white square
        x = int((t / traj_length) * (W - 10))
        frame[50:60, x:x+10, :] = 1.0
        traj3.append(frame)
    
    # Concatenate all trajectories
    all_frames = np.array(traj1 + traj2 + traj3)
    print(f"Created synthetic data: {all_frames.shape}")
    print(f"  - Trajectory 1: frames 0-19 (red)")
    print(f"  - Trajectory 2: frames 20-39 (green)")
    print(f"  - Trajectory 3: frames 40-59 (blue)")
    
    return all_frames


def main():
    print("=" * 70)
    print("Trajectory Boundary Detection Example")
    print("=" * 70)
    
    # Create synthetic data
    observations = create_synthetic_trajectories()
    
    print("\n" + "=" * 70)
    print("1. Detecting Trajectory Boundaries")
    print("=" * 70)
    
    # Test boundary detection with different thresholds
    for threshold in [0.1, 0.2, 0.3]:
        print(f"\n--- Testing with threshold = {threshold} ---")
        boundaries = detect_trajectory_boundaries(observations, threshold)
        detected_indices = np.where(boundaries)[0]
        print(f"Detected boundaries at indices: {detected_indices}")
    
    print("\n" + "=" * 70)
    print("2. Frame Stacking WITHOUT Boundary Detection")
    print("=" * 70)
    print("(This is the OLD behavior - will mix frames across trajectories)")
    
    stacked_no_detection = create_spaced_frame_stack(
        observations,
        frame_stack=4,
        frame_offsets=[0, -1, -2, -3],  # Look back 3 frames
        detect_boundaries=False
    )
    
    # Check frame 20 (first frame of trajectory 2)
    print(f"\nFrame 20 (start of trajectory 2):")
    print(f"  Without detection: This incorrectly includes frames 17, 18, 19 from trajectory 1")
    print(f"  Stacked shape: {stacked_no_detection[20].shape}")
    
    print("\n" + "=" * 70)
    print("3. Frame Stacking WITH Boundary Detection")
    print("=" * 70)
    print("(This is the NEW behavior - properly handles trajectory boundaries)")
    
    stacked_with_detection = create_spaced_frame_stack(
        observations,
        frame_stack=4,
        frame_offsets=[0, -1, -2, -3],  # Look back 3 frames
        detect_boundaries=True,
        boundary_threshold=0.2
    )
    
    print(f"\nFrame 20 (start of trajectory 2):")
    print(f"  With detection: This correctly replicates frame 20 when looking back would cross boundary")
    print(f"  Stacked shape: {stacked_with_detection[20].shape}")
    
    # Compare the difference
    print("\n" + "=" * 70)
    print("4. Verifying the Fix")
    print("=" * 70)
    
    # At trajectory boundaries, the stacked observations should be different
    for boundary_idx in [20, 40]:
        diff = np.mean(np.abs(stacked_no_detection[boundary_idx] - stacked_with_detection[boundary_idx]))
        print(f"\nFrame {boundary_idx} (trajectory boundary):")
        print(f"  Mean difference between methods: {diff:.6f}")
        print(f"  {'✓ Methods produce different results (expected!)' if diff > 0.01 else '✗ Methods are too similar'}")
    
    # Away from boundaries, they should be similar
    for regular_idx in [10, 30, 50]:
        diff = np.mean(np.abs(stacked_no_detection[regular_idx] - stacked_with_detection[regular_idx]))
        print(f"\nFrame {regular_idx} (middle of trajectory):")
        print(f"  Mean difference between methods: {diff:.6f}")
        print(f"  {'✓ Methods produce similar results (expected!)' if diff < 0.01 else '✗ Methods are too different'}")
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
The improved frame stacking function:
1. Automatically detects trajectory boundaries by comparing consecutive frames
2. At trajectory boundaries, replicates the first frame of the new trajectory
   instead of incorrectly pulling frames from the previous trajectory
3. This prevents the model from learning spurious correlations between
   unrelated trajectories

To use in your config:
  config = {
      'detect_trajectory_boundaries': True,      # Enable boundary detection
      'trajectory_boundary_threshold': 0.3,      # Adjust sensitivity (0-1)
      'frame_offsets': [0, -1, -2, -3],         # Or None for consecutive
  }
  
To disable (use old behavior):
  config = {
      'detect_trajectory_boundaries': False,
  }
""")
    print("=" * 70)


if __name__ == '__main__':
    main()

