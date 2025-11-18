# Frame Stacking with Trajectory Boundary Detection

## Problem

The original frame stacking implementation used `jnp.roll()` which wraps around at the edges of arrays. This caused a critical issue at trajectory boundaries: when stacking frames at the start of a new trajectory, it would incorrectly include frames from the end of the previous trajectory.

**Example of the problem:**
```
Dataset: [Traj1_frame1, ..., Traj1_frame50, Traj2_frame1, Traj2_frame2, ...]

Old behavior at Traj2_frame1 with 4-frame stack [0, -1, -2, -3]:
  Stack = [Traj2_frame1, Traj1_frame50, Traj1_frame49, Traj1_frame48]  ❌ WRONG!
  
  This mixes frames from completely different trajectories!
```

## Solution

The improved implementation detects trajectory boundaries by comparing consecutive frames and properly handles frame stacking at boundaries.

### Key Features

1. **Automatic Boundary Detection**: Compares consecutive frames using normalized pixel differences to detect when a new trajectory starts

2. **Smart Frame Replication**: At trajectory boundaries, replicates the first frame of the new trajectory instead of wrapping to the previous trajectory

3. **Configurable Sensitivity**: Adjustable threshold for boundary detection

**New behavior at Traj2_frame1 with 4-frame stack [0, -1, -2, -3]:**
```
Stack = [Traj2_frame1, Traj2_frame1, Traj2_frame1, Traj2_frame1]  ✓ CORRECT!

As the trajectory progresses:
Traj2_frame2: [Traj2_frame2, Traj2_frame1, Traj2_frame1, Traj2_frame1]
Traj2_frame3: [Traj2_frame3, Traj2_frame2, Traj2_frame1, Traj2_frame1]
Traj2_frame4: [Traj2_frame4, Traj2_frame3, Traj2_frame2, Traj2_frame1]
Traj2_frame5: [Traj2_frame5, Traj2_frame4, Traj2_frame3, Traj2_frame2]  ✓ Normal stacking
```

## Usage

### Basic Usage (Enabled by Default)

The boundary detection is **enabled by default** with a threshold of 0.3:

```python
from train import load_dataset

config = {
    'frame_stack': 4,
    # Boundary detection is enabled by default
}

dataset = load_dataset(data, frame_stack=4, config=config)
```

### Custom Configuration

```python
config = {
    'frame_stack': 4,
    'frame_offsets': [0, -1, -2, -3],          # Consecutive frames (default)
    # OR
    'frame_offsets': [0, -5, -10, -15],        # Spaced frames
    
    # Boundary detection settings
    'detect_trajectory_boundaries': True,       # Enable detection (default: True)
    'trajectory_boundary_threshold': 0.3,       # Threshold (default: 0.3)
}
```

### Threshold Tuning

The `trajectory_boundary_threshold` parameter controls sensitivity:

- **Lower values (0.1 - 0.2)**: More sensitive
  - Detects smaller scene changes
  - May split continuous trajectories if there are sudden movements
  - Use for datasets with very distinct trajectories

- **Medium values (0.3 - 0.4)**: Balanced (recommended)
  - Good for most use cases
  - Detects major scene changes
  - Default: 0.3

- **Higher values (0.5 - 0.7)**: Less sensitive
  - Only detects very large scene changes
  - May miss some trajectory boundaries
  - Use if trajectories have major visual variations within them

### Disable Boundary Detection (Old Behavior)

If you want to use the old behavior (not recommended):

```python
config = {
    'detect_trajectory_boundaries': False,
}
```

## Example Script

Run the example to see the difference:

```bash
python trajectory_boundary_detection_example.py
```

This script:
1. Creates synthetic video data with 3 distinct trajectories
2. Shows boundary detection at different thresholds
3. Compares old vs new frame stacking behavior
4. Verifies the fix at trajectory boundaries

## Implementation Details

### Boundary Detection Function

```python
def detect_trajectory_boundaries(observations, threshold=0.3):
    """
    Detect trajectory boundaries by comparing consecutive frames.
    
    Returns a boolean array where True indicates the start of a new trajectory.
    """
    # Compute mean absolute difference between consecutive frames
    diffs = mean_abs_diff(observations[1:], observations[:-1])
    
    # Mark frames with large differences as trajectory starts
    is_trajectory_start = diffs > threshold
    is_trajectory_start[0] = True  # First frame is always a start
    
    return is_trajectory_start
```

### Frame Stacking Logic

For each frame at time `t`:
1. Find the start index of the current trajectory
2. For each frame offset (e.g., -1, -2, -3):
   - Calculate target index: `target = t + offset`
   - If `target < trajectory_start`:
     - Use the first frame of the current trajectory (replicate)
   - Else:
     - Use the frame at `target` (normal stacking)
3. Concatenate all frames along the channel dimension

## Benefits

1. **Prevents Data Leakage**: No mixing of frames from different trajectories
2. **Better Training**: Model doesn't learn spurious correlations between unrelated scenes
3. **Automatic**: No manual trajectory annotations needed
4. **Robust**: Works with any dataset that has visual scene changes between trajectories
5. **Configurable**: Adjustable sensitivity for different use cases

## When to Adjust Settings

### Increase Threshold (0.4 - 0.6) if:
- Your trajectories have large visual variations (camera angle changes, lighting changes)
- You see too many detected boundaries in the middle of continuous trajectories
- Training seems unstable due to oversegmentation

### Decrease Threshold (0.1 - 0.2) if:
- Your trajectories are very similar visually
- Scene transitions are subtle
- You notice the model is still seeing mixed trajectories

### Disable Detection if:
- Your dataset is already properly segmented with all trajectories in separate chunks
- You're using very short frame stacks (frame_stack < 3)
- Your trajectories never have abrupt scene changes

## Technical Notes

- The boundary detection adds minimal computational overhead (< 1% of loading time)
- Detection is done once during dataset loading, not during training
- The threshold is applied to normalized pixel differences (0-1 range)
- Frame replication at boundaries is a common technique in video processing
- This approach is similar to padding strategies in temporal convolution networks

## Visualization Script

A separate script `visualize_dataset.py` is available to stream and visualize dataset observations in wandb:

```bash
# Visualize a dataset by path
python visualize_dataset.py --dataset_path /path/to/dataset.npz

# Or by dataset name (auto-downloads if needed)
python visualize_dataset.py --dataset_name visual-cube-lift-v0
```

This helps verify that:
1. Trajectories are properly detected
2. No frame mixing occurs at boundaries
3. Frame stacking looks correct throughout trajectories

