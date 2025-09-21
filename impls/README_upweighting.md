# Action Upweighting for Behavior Encouragement

This feature allows you to upweight specific action types during training to encourage certain behaviors, such as turning at stoplights or lane changes.

## Overview

The training system automatically labels actions into 5 discrete categories:
- **0: straight** - Going straight ahead
- **1: left turn** - Sharp left turns
- **2: right turn** - Sharp right turns  
- **3: left lane change** - Gradual left lane changes
- **4: right lane change** - Gradual right lane changes

By upweighting specific action types, you can make the model see more examples of those behaviors during training, encouraging it to learn and reproduce them more effectively.

## Usage

### 1. Upweight Turns (Recommended for Stoplight Turning)

To encourage turning behavior (useful for stoplights, intersections):

```bash
python train2.py --dataset_path your_data.npz --upweight_turns --turn_weight 3.0
```

This will upweight both left turns (action 1) and right turns (action 2) by 3x their normal frequency.

### 2. Upweight Lane Changes

To encourage lane changing behavior:

```bash
python train2.py --dataset_path your_data.npz --upweight_lane_changes --lane_change_weight 2.0
```

This will upweight both left lane changes (action 3) and right lane changes (action 4) by 2x.

### 3. Custom Weights

For fine-grained control, you can specify custom weights for each action type:

```bash
python train2.py --dataset_path your_data.npz --custom_weights "1:5.0" "2:5.0" "3:2.0"
```

This upweights:
- Left turns (1) by 5x
- Right turns (2) by 5x  
- Left lane changes (3) by 2x

### 4. Combine with Guided Training

You can combine upweighting with guided training for even better results:

```bash
python train2.py --dataset_path your_data.npz --upweight_turns --turn_weight 3.0 --use_guided
```

## Using the Convenience Script

For easier usage, you can use the `train_with_upweighting.py` script:

```bash
# Upweight turns by 3x
python train_with_upweighting.py --dataset_path your_data.npz --upweight_turns --turn_weight 3.0

# Upweight lane changes by 2x
python train_with_upweighting.py --dataset_path your_data.npz --upweight_lane_changes --lane_change_weight 2.0

# Custom weights
python train_with_upweighting.py --dataset_path your_data.npz --custom_weights "1:5.0" "2:5.0"

# Combine with guided training
python train_with_upweighting.py --dataset_path your_data.npz --upweight_turns --turn_weight 3.0 --use_guided
```

## Monitoring

The training will log action distributions to help you verify the upweighting is working:

- **Console output**: Every 10,000 steps shows the action distribution in the current batch
- **Wandb logs**: Action distributions are logged every `log_every` steps under `action_dist/`

Example output:
```
Action distribution: {'straight': 0.4, 'left_turn': 0.3, 'right_turn': 0.3, 'left_lane_change': 0.0, 'right_lane_change': 0.0}
```

## Tips for Effective Upweighting

1. **Start Moderate**: Begin with weights of 2-3x to avoid overfitting to specific actions
2. **Monitor Performance**: Watch the action distribution logs to ensure the upweighting is working
3. **Combine Strategies**: Use guided training (`--use_guided`) with upweighting for best results
4. **Experiment**: Try different weight values to find what works best for your dataset
5. **Balance**: Don't upweight too aggressively - you still want the model to learn all behaviors

## Example: Encouraging Stoplight Turning

For a driving dataset where you want to encourage turning at stoplights:

```bash
# Option 1: Upweight all turns
python train2.py --dataset_path driving_data.npz --upweight_turns --turn_weight 3.0 --use_guided

# Option 2: Custom weights for more control
python train2.py --dataset_path driving_data.npz --custom_weights "1:4.0" "2:4.0" --use_guided

# Option 3: Also upweight lane changes for more complex maneuvers
python train2.py --dataset_path driving_data.npz --custom_weights "1:3.0" "2:3.0" "3:2.0" "4:2.0" --use_guided
```

## Technical Details

- **Sampling Method**: Uses weighted random sampling with replacement
- **Weight Normalization**: Weights are automatically normalized to maintain proper sampling
- **Backward Compatibility**: The original `sample_batch` function still works without weights
- **Memory Efficient**: Weights are computed on-the-fly without storing additional data

## Troubleshooting

**Q: The action distribution doesn't change much**
A: Try increasing the weight values or check if your dataset has enough examples of the target actions

**Q: Training becomes unstable**
A: Reduce the weight values - too much upweighting can cause overfitting

**Q: I want to upweight different actions**
A: Use the `--custom_weights` option to specify exact action:weight pairs

**Q: How do I know what weight to use?**
A: Start with 2-3x and monitor the action distribution logs. Adjust based on your desired balance. 