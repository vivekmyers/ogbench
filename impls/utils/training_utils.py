"""Training utilities for learning rate scheduling and data preprocessing."""

import optax
import numpy as np


def create_lr_schedule(initial_lr, decay_type, decay_steps, decay_rate, warmup_steps, total_steps):
    """Create a learning rate schedule function.
    
    Args:
        initial_lr: Initial learning rate
        decay_type: 'none', 'linear', 'cosine', 'exponential'
        decay_steps: Steps over which to decay (for linear/cosine) or decay period (for exponential)
        decay_rate: Final LR as fraction of initial (for linear/cosine) or decay rate (for exponential)
        warmup_steps: Warmup steps before decay starts
        total_steps: Total training steps (for cosine schedule)
    
    Returns:
        A function that takes step count and returns learning rate
    """
    if decay_type == 'none' or decay_steps <= 0:
        return lambda step: initial_lr
    
    # Warmup schedule
    if warmup_steps > 0:
        warmup_schedule = optax.linear_schedule(
            init_value=0.0,
            end_value=initial_lr,
            transition_steps=warmup_steps
        )
    else:
        warmup_schedule = lambda step: initial_lr
    
    # Decay schedule (starts after warmup)
    if decay_type == 'linear':
        decay_schedule = optax.linear_schedule(
            init_value=initial_lr,
            end_value=initial_lr * decay_rate,
            transition_steps=decay_steps
        )
        def schedule(step):
            if step < warmup_steps:
                return warmup_schedule(step)
            else:
                decay_step = step - warmup_steps
                return decay_schedule(decay_step) if decay_step < decay_steps else initial_lr * decay_rate
        return schedule
    
    elif decay_type == 'cosine':
        decay_schedule = optax.cosine_decay_schedule(
            init_value=initial_lr,
            decay_steps=decay_steps,
            alpha=decay_rate  # Final value as fraction of initial
        )
        def schedule(step):
            if step < warmup_steps:
                return warmup_schedule(step)
            else:
                decay_step = step - warmup_steps
                return decay_schedule(decay_step) if decay_step < decay_steps else initial_lr * decay_rate
        return schedule
    
    elif decay_type == 'exponential':
        def schedule(step):
            if step < warmup_steps:
                return warmup_schedule(step)
            else:
                decay_step = step - warmup_steps
                # Exponential decay: lr = initial_lr * (decay_rate ^ (decay_step / decay_steps))
                num_decays = decay_step // decay_steps
                return initial_lr * (decay_rate ** num_decays)
        return schedule
    
    else:
        return lambda step: initial_lr


def continuous_to_discrete_bins_numpy(actions: np.ndarray, num_bins: int = 32) -> np.ndarray:
    """Convert continuous actions to discrete bin indices (numpy version for preprocessing).
    
    Args:
        actions: Continuous actions of shape (..., 3) where:
            actions[..., 0] = throttle in [0, 1]
            actions[..., 1] = steer in [-1, 1]
            actions[..., 2] = brake in [0, 1]
        num_bins: Number of bins per dimension (default: 32)
    
    Returns:
        Discrete bin indices of shape (..., 3) with values in [0, num_bins-1]
    """
    # Clip actions to valid ranges before binning
    throttle = np.clip(actions[..., 0], 0.0, 1.0)  # [0, 1]
    steer = np.clip(actions[..., 1], -1.0, 1.0)    # [-1, 1]
    brake = np.clip(actions[..., 2], 0.0, 1.0)     # [0, 1]
    
    # Convert to [0, 1] range for steer
    steer_normalized = (steer + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    
    # Convert to bin indices: [0, 1] -> [0, num_bins-1]
    throttle_bins = np.minimum(np.floor(throttle * num_bins), num_bins - 1).astype(np.int32)
    steer_bins = np.minimum(np.floor(steer_normalized * num_bins), num_bins - 1).astype(np.int32)
    brake_bins = np.minimum(np.floor(brake * num_bins), num_bins - 1).astype(np.int32)
    
    return np.stack([throttle_bins, steer_bins, brake_bins], axis=-1)

