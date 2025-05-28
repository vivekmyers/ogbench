#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange
import wandb
import flax.serialization as fxs

from agents.crl import CRLAgent, get_config  # Correct import for CRL agent


# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------

def load_dataset(path: str | Path, frame_stack: int = 1) -> Dict[str, jnp.ndarray]:
    """Load an .npz into device arrays, with optional frame stacking.
    
    Args:
        path: Path to the .npz dataset
        frame_stack: Number of frames to stack. Default is 1 (no stacking).
    """
    data_np = np.load(path)
    # Create a minimal dataset with required fields
    dataset = {k: jnp.asarray(v) for k, v in data_np.items()}
    
    # IMPORTANT: Normalize image observations right at the start if needed
    if (
        "observations" in dataset 
        and len(dataset["observations"].shape) == 4  # [batch, height, width, channels]
    ):
        # Check if normalization is needed based on data type or values
        obs_max = float(jnp.max(dataset["observations"]))
        needs_normalization = dataset["observations"].dtype in [np.uint8, jnp.uint8] or obs_max > 1.0
        
        if needs_normalization:
            print(f"Normalizing image observations: dtype={dataset['observations'].dtype}, max_value={obs_max}")
            # Convert to float32 and normalize to [0, 1]
            dataset["observations"] = dataset["observations"].astype(jnp.float32) / 255.0
            
            # Normalize next_observations if present
            if "next_observations" in dataset:
                dataset["next_observations"] = dataset["next_observations"].astype(jnp.float32) / 255.0
    
    # Add terminals if missing (CRITICAL for CRL to work)
    if "terminals" not in dataset:
        print("No terminals found in dataset, marking last state as terminal")
        dataset["terminals"] = jnp.zeros(dataset["observations"].shape[0], dtype=jnp.bool_)
        dataset["terminals"] = dataset["terminals"].at[-1].set(True)
    else:
        # Check if there are any terminal states, if not mark the last one
        if not jnp.any(dataset["terminals"]):
            print("No terminal states in dataset, marking last state as terminal")
            dataset["terminals"] = dataset["terminals"].at[-1].set(True)
    
    print(f"Dataset loaded: {len(dataset['observations'])} transitions")
    print(f"Terminal states: {jnp.sum(dataset['terminals'])}")
    print(f"Observation shape: {dataset['observations'].shape}, dtype: {dataset['observations'].dtype}")
    print(f"Observation min: {float(jnp.min(dataset['observations']))}, max: {float(jnp.max(dataset['observations']))}")
    
    # Apply frame stacking if requested
    if frame_stack > 1:
        print(f"Applying frame stacking with {frame_stack} frames")
        
        # Get original observation shape
        original_obs = dataset["observations"]
        n_transitions = original_obs.shape[0]
        
        # For image observations
        if len(original_obs.shape) == 4:  # [batch, height, width, channels]
            h, w, c = original_obs.shape[1:]
            
            # Create stacked observations
            stacked_shape = (n_transitions, h, w, c * frame_stack)
            stacked_obs = np.zeros(stacked_shape, dtype=original_obs.dtype)
            
            # Fill stacked observations
            for i in range(n_transitions):
                # For each position, get up to frame_stack previous frames
                for j in range(frame_stack):
                    # Get frame, respecting trajectory boundaries using terminals
                    frame_idx = max(0, i - j)
                    # Don't cross episode boundaries
                    if j > 0 and i >= j:
                        # Check if any terminal states exist between frame_idx and i
                        if jnp.any(dataset["terminals"][frame_idx:i]):
                            # If so, just repeat the current frame
                            frame_idx = i
                    
                    # Place the frame in the stack (in reverse order so most recent is last)
                    stack_idx = frame_stack - 1 - j
                    stacked_obs[i, :, :, stack_idx * c:(stack_idx + 1) * c] = original_obs[frame_idx]
            
            # Replace observations with stacked version
            dataset["observations"] = jnp.asarray(stacked_obs)
            
            # For next_observations, shift by 1 and handle terminal states
            next_stacked_obs = np.zeros(stacked_shape, dtype=original_obs.dtype)
            for i in range(n_transitions):
                next_idx = min(i + 1, n_transitions - 1)
                # If terminal, next_obs is a duplicate of the current obs
                if i < n_transitions - 1 and dataset["terminals"][i]:
                    next_stacked_obs[i] = stacked_obs[i]
                else:
                    next_stacked_obs[i] = stacked_obs[next_idx]
                    
            dataset["next_observations"] = jnp.asarray(next_stacked_obs)
            
            print(f"Stacked observation shape: {dataset['observations'].shape}")
            print(f"Stacked observation min: {float(jnp.min(dataset['observations']))}, max: {float(jnp.max(dataset['observations']))}")
        else:
            print(f"Warning: Frame stacking only implemented for image observations. Shape: {original_obs.shape}")
    
    return dataset


def sample_batch(dataset: Dict[str, jnp.ndarray], *, batch_size: int, key: jax.random.PRNGKey) -> tuple[Dict[str, jnp.ndarray], jax.random.PRNGKey]:
    """Minimal trajectory-aware sampler for CRL."""
    n = dataset["observations"].shape[0]
    key, subkey = jax.random.split(key)
    idx = jax.random.choice(subkey, n, shape=(batch_size,), replace=True)
    
    # Find terminal indices to segment trajectories
    terminal_indices = jnp.where(dataset["terminals"])[0]
    
    # If no terminals found, treat the last state as terminal
    if len(terminal_indices) == 0:
        print("No terminal states found for sampling, treating the last state as terminal")
        terminal_indices = jnp.array([n - 1])
    
    # Sample goals using basic trajectory awareness
    batch = {
        "observations": dataset["observations"][idx],
        "actions": dataset["actions"][idx],
        "next_observations": dataset["observations"][jnp.minimum(idx + 1, n - 1)],
        "rewards": jnp.zeros((batch_size,)),  # Will be computed based on goals
        "masks": jnp.ones((batch_size,)),     # Will be computed based on goals
    }
    
    # Handle next observations with special care if frame stacking is used
    # If the dataset has pre-computed next_observations, use those
    if "next_observations" in dataset:
        batch["next_observations"] = dataset["next_observations"][idx]
    
    # Add goals (using future states in the same trajectory 50% of the time)
    key, goal_key = jax.random.split(key)
    use_future = jax.random.uniform(goal_key, (batch_size,)) < 0.5
    
    # For each index, find the nearest terminal
    traj_ends = jnp.searchsorted(terminal_indices, idx)
    future_idx = jnp.minimum(idx + 1, terminal_indices[jnp.minimum(traj_ends, len(terminal_indices) - 1)])
    
    # Random goals
    key, random_key = jax.random.split(key)
    random_idx = jax.random.choice(random_key, n, shape=(batch_size,), replace=True)
    
    # Combine
    goal_idx = jnp.where(use_future, future_idx, random_idx)
    batch["value_goals"] = dataset["observations"][goal_idx]
    batch["actor_goals"] = dataset["observations"][goal_idx]
    
    # Goal-conditioned rewards
    batch["rewards"] = jnp.full((batch_size,), -1.0)  # -1 everywhere except at goal
    batch["masks"] = jnp.ones((batch_size,))          # 1 everywhere except at goal
    
    # For image observations, compute a normalized L2 distance
    if len(batch["next_observations"].shape) > 2:  # Image observations
        # Flatten images to compute L2 distance
        next_obs_flat = batch["next_observations"].reshape(batch_size, -1)
        goals_flat = batch["value_goals"].reshape(batch_size, -1)
        
        # Compute L2 distance and normalize by the number of elements
        distances = jnp.sqrt(jnp.sum((next_obs_flat - goals_flat)**2, axis=1)) / jnp.sqrt(next_obs_flat.shape[1])
        
        # Use percentile-based threshold instead of fixed threshold
        # Sort distances to find the threshold value at a given percentile (lowest 5%)
        target_goal_rate = 0.05  # Target 5% of states as goals
        
        # Sort distances and find threshold
        sorted_distances = jnp.sort(distances)
        percentile_idx = int(target_goal_rate * batch_size)
        if percentile_idx > 0:
            # Use the distance at the desired percentile as threshold
            goal_threshold = sorted_distances[percentile_idx]
        else:
            # Fallback if batch is too small
            goal_threshold = jnp.min(distances) + 0.000001  # Just above minimum
        
        # Use this threshold to identify goals
        is_at_goal = distances <= goal_threshold
        
        # Print stats about goal detection periodically
        key, debug_key = jax.random.split(key)
        if jax.random.uniform(debug_key) < 0.001:  # ~0.1% chance to print stats
            print(f"Goal distance stats: min={float(jnp.min(distances)):.6f}, "
                  f"mean={float(jnp.mean(distances)):.6f}, "
                  f"max={float(jnp.max(distances)):.6f}, "
                  f"threshold={float(goal_threshold):.6f}, "
                  f"at_goal_rate={float(jnp.mean(is_at_goal)):.6f}")
    else:  # Vector observations
        # For vector observations, use a simpler L2 distance with percentile threshold
        distances = jnp.sqrt(jnp.sum((batch["next_observations"] - batch["value_goals"])**2, axis=1))
        
        # Sort distances to find the threshold value at a given percentile (lowest 5%)
        sorted_distances = jnp.sort(distances)
        percentile_idx = int(0.05 * batch_size)  # Target 5% of states as goals
        goal_threshold = sorted_distances[percentile_idx] if percentile_idx > 0 else jnp.min(distances) + 0.000001
        
        # Use this threshold to identify goals
        is_at_goal = distances <= goal_threshold
    
    # Generate smoother rewards based on distance (linear decay)
    # Use the threshold to scale rewards from -1 to 0
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-6
    normalized_distances = distances / (goal_threshold + epsilon)
    smooth_rewards = jnp.clip(-normalized_distances, -1.0, 0.0)
    
    # Use smoothed rewards for learning
    batch["rewards"] = smooth_rewards
    
    # Set mask to 0 for transitions that reach the goal (terminal)
    batch["masks"] = jnp.where(is_at_goal, 0.0, batch["masks"])
    
    return batch, key


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    # ---------------- WandB ----------------
    cfg = get_config()
    cfg.batch_size = args.batch_size
    cfg.actor_loss = args.actor_loss
    cfg.encoder = "impala"  # Use impala encoder instead of resnet18
    
    # Add minimum CRL goal params with better goal sampling for critic
    # More weight on trajectory-based goals (essential for contrastive learning)
    cfg.value_p_curgoal = 0.0
    cfg.value_p_trajgoal = 0.5  # Equal probability for trajectory and random goals
    cfg.value_p_randomgoal = 0.5  # Equal probability for trajectory and random goals
    
    # Actor goal sampling can remain more balanced
    cfg.actor_p_curgoal = 0.0
    cfg.actor_p_trajgoal = 0.5
    cfg.actor_p_randomgoal = 0.5
    
    # These affect how future goals are sampled
    cfg.value_geom_sample = True  # Sample goals geometrically (closer goals more likely)
    cfg.actor_geom_sample = True
    cfg.gc_negative = True  # Use -1 reward for non-goal states
    
    # Tuned parameters for better learning
    cfg.temperature = 0.2  # Lower temperature helps with sharper logits for contrastive learning
    cfg.lr = 1e-4  # Slightly reduced learning rate for stability
    cfg.replay_size = 1000000  # Large replay buffer capacity
    cfg.target_update_interval = 1  # Frequent target updates for faster value propagation
    cfg.critic_train_freq = 1  # Train critic every step
    cfg.latent_dim = 256  # Sufficient latent dimension for representation
    
    # A higher discount factor is often better for goal-conditioned tasks
    # since we want to propagate the reward signal more effectively
    cfg.discount = 0.99
    
    # Increase batch size if not already set
    if cfg.batch_size < 1024:
        cfg.batch_size = 1024  # Larger batch size improves goal statistics
    
    wandb.init(project=args.project, config=cfg.to_dict())

    # ---------------- Data -----------------
    dataset = load_dataset(args.dataset_path, frame_stack=args.frame_stack)
    # Normalization is now handled in load_dataset

    # ---------------- Agent ---------------
    agent = CRLAgent.create(
        seed=args.seed,
        ex_observations=dataset["observations"][0],
        ex_actions=dataset["actions"][0],
        config=cfg,
    )

    rng = jax.random.PRNGKey(args.seed + 1)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    progress = trange(args.steps, dynamic_ncols=True)
    
    # Add tracking of critic performance
    critic_losses = []
    reward_stats = []
    
    for step in progress:
        rng, batch_key = jax.random.split(rng)
        batch, rng = sample_batch(dataset, batch_size=cfg.batch_size, key=batch_key)
        
        # Save stats about the batch for debugging
        reward_stats.append({
            "mean": float(jnp.mean(batch["rewards"])),
            "min": float(jnp.min(batch["rewards"])),
            "max": float(jnp.max(batch["rewards"])),
            "at_goal": float(jnp.mean(batch["masks"] == 0.0))
        })
        
        # Print batch stats occasionally to help with debugging
        if step % 1000 == 0:
            print(f"Step {step}: at_goal_rate={float(jnp.mean(batch['masks'] == 0.0)):.4f}, "
                  f"reward_mean={float(jnp.mean(batch['rewards'])):.4f}, "
                  f"reward_min={float(jnp.min(batch['rewards'])):.4f}, "
                  f"reward_max={float(jnp.max(batch['rewards'])):.4f}")
        
        agent, info = agent.update(batch)
        
        # Track critic loss
        critic_losses.append(float(info.get("critic/contrastive_loss", 0.0)))

        # ---- Logging ----
        if step % args.log_every == 0:
            # Base metrics
            log_dict = {k: float(v) for k, v in info.items()}
            
            # Add goal and reward statistics
            if reward_stats:
                # Average over recent steps
                recent_rewards = reward_stats[-args.log_every:]
                log_dict.update({
                    "rewards/mean": np.mean([s["mean"] for s in recent_rewards]),
                    "rewards/min": np.min([s["min"] for s in recent_rewards]),
                    "rewards/max": np.max([s["max"] for s in recent_rewards]),
                    "rewards/at_goal_rate": np.mean([s["at_goal"] for s in recent_rewards])
                })
            
            # Add critic loss tracking
            if critic_losses:
                recent_losses = critic_losses[-args.log_every:]
                log_dict["critic/loss_std"] = np.std(recent_losses)
                log_dict["critic/loss_change"] = (recent_losses[-1] - recent_losses[0]) if len(recent_losses) > 1 else 0
            
            wandb.log(log_dict, step=step)
            
            # Update progress bar with more info
            progress.set_postfix(
                critic_loss=float(info.get("critic/contrastive_loss", 0.0)),
                reward_mean=reward_stats[-1]["mean"] if reward_stats else 0,
                at_goal=reward_stats[-1]["at_goal"] if reward_stats else 0
            )

        # ---- Checkpoint ----
        if args.ckpt_every and step and step % args.ckpt_every == 0:
            ckpt_path = ckpt_dir / f"agent_step{step}.pkl"
            with ckpt_path.open("wb") as f:
                f.write(fxs.to_bytes(agent))
    
    # Save final model
    final_model_path = ckpt_dir / "final_model.pkl"
    print(f"Saving final model to {final_model_path}")
    with final_model_path.open("wb") as f:
        f.write(fxs.to_bytes(agent))

    wandb.finish()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal CRL offline training script")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to .npz offline dataset")
    parser.add_argument("--steps", type=int, default=700_000, help="Total gradient steps")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--actor_loss", choices=["awr", "ddpgbc"], default="ddpgbc")
    parser.add_argument("--project", default="crl_offline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=10_000)
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--frame_stack", type=int, default=1, help="Number of frames to stack (default: 1, no stacking)")
    args = parser.parse_args()
    main(args)
