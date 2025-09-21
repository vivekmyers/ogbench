from __future__ import annotations

import argparse
import time
from importlib import import_module
from pathlib import Path
from typing import Dict, Sequence

import flax.serialization as fxs
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from tqdm import trange

import distrax



def load_dataset(dataset: Dict[str, np.ndarray], frame_stack: int = 1, chunk_size: int = 20000, index: int = 0, config: dict = None) -> Dict[str, jnp.ndarray]:
    start = time.time()
    print(dataset['observations'].shape, dataset['observations'].size)
    # Create a copy to avoid modifying the original dataset
    print("Copying chunk data...")
    chunk_data = {
        'observations': dataset['observations'][index:(index + chunk_size)].copy(),
        #'terminals': dataset['terminals'][index:(index + chunk_size)].copy(),
        'terminals': np.zeros(chunk_size, dtype=np.bool_),
        'actions': dataset['actions'][index:(index + chunk_size)].copy(),
    }

    # Discretize actions if discrete flag is set
    if config.get('discrete', False):
        print("Discretizing continuous actions using independent softmax (32 bins per dimension, 3D discrete actions)...")
        # Get action components
        throttle = chunk_data['actions'][:, 0]  # 0 to 1
        steer = chunk_data['actions'][:, 1]     # -1 to 1
        brake = chunk_data['actions'][:, 2] if chunk_data['actions'].shape[1] > 2 else jnp.zeros_like(throttle)  # 0 to 1
        # Use 32 bins for each dimension
        num_bins = 32
        # Create bin centers for each dimension
        throttle_centers = jnp.linspace(0, 1, num_bins)  # [0, 1/31, 2/31, ..., 1]
        steer_centers = jnp.linspace(-1, 1, num_bins)    # [-1, -29/31, -27/31, ..., 1]
        brake_centers = jnp.linspace(0, 1, num_bins)     # [0, 1/31, 2/31, ..., 1]
        throttle_distances = jnp.abs(throttle[:, None] - throttle_centers[None, :])
        steer_distances = jnp.abs(steer[:, None] - steer_centers[None, :])
        brake_distances = jnp.abs(brake[:, None] - brake_centers[None, :])
        
        # Convert distances to logits (negative distances = higher logits)
        temperature = 0.7  # Much lower temperature for smoother distributions
        throttle_logits = -throttle_distances * temperature
        steer_logits = -steer_distances * temperature
        brake_logits = -brake_distances * temperature
        
        # Clip logits to prevent extreme values that cause numerical instability
        max_logit = 5.0  # More aggressive clipping
        min_logit = -5.0  # More aggressive clipping
        
        throttle_logits = jnp.clip(throttle_logits, min_logit, max_logit)
        steer_logits = jnp.clip(steer_logits, min_logit, max_logit)
        brake_logits = jnp.clip(brake_logits, min_logit, max_logit)
        
        # Apply softmax to get probability distributions over bins
        throttle_probs = jax.nn.softmax(throttle_logits, axis=-1)
        steer_probs = jax.nn.softmax(steer_logits, axis=-1)
        brake_probs = jax.nn.softmax(brake_logits, axis=-1)
        
        # For training, we still need discrete actions, so use the mode (most likely bin)
        throttle_discrete = jnp.argmax(throttle_probs, axis=-1)
        steer_discrete = jnp.argmax(steer_probs, axis=-1)
        brake_discrete = jnp.argmax(brake_probs, axis=-1)
        
        # Store the discrete actions as a 3D vector
        discrete_actions = jnp.stack([throttle_discrete, steer_discrete, brake_discrete], axis=-1)
        
        # Replace continuous actions with discrete version
        chunk_data['actions'] = discrete_actions
        
        # Store the soft distributions for potential use in loss computation
        chunk_data['throttle_probs'] = throttle_probs
        chunk_data['steer_probs'] = steer_probs
        chunk_data['brake_probs'] = brake_probs
        
        print(f"Action space discretized using independent softmax. Shape: {discrete_actions.shape}")
        print(f"Total discrete actions per dimension: {num_bins} (throttle, steer, brake)")
        print(f"Softmax temperature: {temperature}")
    
    # Copy FULL mapping arrays (not sliced!) - they map global positions, not chunk positions
    if 'original_to_shuffled' in dataset:
        chunk_data['original_to_shuffled'] = dataset['original_to_shuffled'].copy()
    if 'shuffled_to_original' in dataset:
        chunk_data['shuffled_to_original'] = dataset['shuffled_to_original'].copy()
    
    n_transitions = len(chunk_data['observations'])
    
    # Safety check: ensure we have data
    if n_transitions == 0:
        raise ValueError(f"Empty chunk at index {index}, chunk_size {chunk_size}, total dataset size {len(dataset['observations'])}")
    h, w, c, = 100, 100, 3

    # Debug original data
    print(f"Original obs stats: min={np.min(chunk_data['observations']):.4f}, max={np.max(chunk_data['observations']):.4f}, mean={np.mean(chunk_data['observations']):.4f}")
    
    # Normalize observations to [0,1] range
    print("Normalizing observations...")
    chunk_data['observations'] = chunk_data['observations'].astype(np.float16) / 255.0
    if not config['discrete']:
        chunk_data['actions'] = chunk_data['actions'].astype(np.float16)
    
    # Debug normalized data
    print(f"Normalized obs stats: min={np.min(chunk_data['observations']):.4f}, max={np.max(chunk_data['observations']):.4f}, mean={np.mean(chunk_data['observations']):.4f}")

    print('Converting to jnp...')
    chunk_data = {k: jnp.array(v) for k, v in chunk_data.items()}

    original_obs = chunk_data['observations'].reshape(n_transitions, h, w, c).astype(np.float32)
    print(original_obs.shape)

    terminals = chunk_data['terminals'][:-3]
    terminals = terminals.at[-1].set(True)
    chunk_data['terminals'] = terminals

    print(f"Creating frame-stacked observations (frame_stack={frame_stack})...")
    shifted_dataset = [jnp.roll(original_obs, shift=i, axis=0) for i in range(frame_stack - 1, -1, -1)]
    concat_dataset = jnp.concatenate(shifted_dataset, axis=-1)
    concat_dataset = concat_dataset[(frame_stack - 1) :]
    chunk_data['observations'] = concat_dataset
    next_obs = jnp.roll(concat_dataset, shift=-1, axis=0)
    next_obs = next_obs.at[-1].set(next_obs[-2])
    chunk_data['next_observations'] = next_obs

    end = time.time()
    return chunk_data

def shuffle_dataset(path: str | Path) -> Dict[str, np.ndarray]:
    print(f"Loading dataset from {path}...")
    data_np = np.load(path)
    print("Dataset loaded")
    start_time = time.time()
    dataset = {k: np.asarray(v, dtype=np.float16) for k, v in data_np.items()}
    end_time = time.time()
    print(f"Dataset conversion took {(end_time - start_time)*1000:.1f}ms")

    steer_values = dataset['actions'][:, 1]

    print(f'Dataset loaded: {len(dataset["observations"])} transitions')
    print(f'Terminal states: {np.sum(dataset["terminals"])}')
    print(f'Observation shape: {dataset["observations"].shape}, dtype: {dataset["observations"].dtype}')

    # # Create bidirectional mapping for efficient goal sampling
    # n_samples = len(dataset['observations'])
    # print(f"Creating shuffled indices for {n_samples} samples...")
    # rng = np.random.default_rng()
    # shuffled_indices = rng.permutation(n_samples)

    # print("Building bidirectional mappings...")
    # # original_to_shuffled[original_pos] = shuffled_pos
    # # "Where did original position i end up after shuffling?"
    # dataset['original_to_shuffled'] = np.zeros(n_samples, dtype=int)
    # for original_pos, shuffled_pos in enumerate(shuffled_indices):
    #     dataset['original_to_shuffled'][original_pos] = shuffled_pos
    
    # # shuffled_to_original[shuffled_pos] = original_pos  
    # # "What was the original position of the frame now at shuffled position i?"
    # dataset['shuffled_to_original'] = np.zeros(n_samples, dtype=int)
    # for original_pos, shuffled_pos in enumerate(shuffled_indices):
    #     dataset['shuffled_to_original'][shuffled_pos] = original_pos
    
    # print("Shuffling dataset arrays...")
    # dataset['observations'] = dataset['observations'][shuffled_indices]
    # dataset['terminals'] = dataset['terminals'][shuffled_indices] 
    # dataset['actions'] = dataset['actions'][shuffled_indices]

    print("Dataset shuffled successfully!")
    return dataset


def compute_validation_loss(agent, val_dataset, sample_batch_fn, batch_size=1000, frame_stack=1, config=None):
    """Compute validation loss on a subset of validation data."""
    
    # Use the same sample_batch function as training for perfect consistency
    val_batch = sample_batch_fn(
        val_dataset,
        batch_size=batch_size,
        frame_stack=frame_stack,
        config=config
    )
    
    try:
        actor_loss, actor_info = agent.actor_loss(val_batch, agent.network.params)
        actor_dist = agent.network.select('actor')(
            val_batch['observations'],
            val_batch['actor_goals'],
            params=agent.network.params
        )
        
        # Handle discrete vs continuous actions properly
        if config.get('discrete', False):
            predicted_actions = actor_dist.mode()
            expert_actions = val_batch['actions']
            
            if config.get('multi_discrete', False):
                # For multi-discrete actions (3D), compute accuracy per dimension
                throttle_accuracy = jnp.mean(predicted_actions[:, 0] == expert_actions[:, 0])
                steer_accuracy = jnp.mean(predicted_actions[:, 1] == expert_actions[:, 1])
                brake_accuracy = jnp.mean(predicted_actions[:, 2] == expert_actions[:, 2])
                
                # Overall exact match accuracy (all dimensions must match)
                exact_match_accuracy = jnp.mean(jnp.all(predicted_actions == expert_actions, axis=-1))
                
                # Compute categorical accuracy for discrete actions
                categorical_accuracy = (throttle_accuracy + steer_accuracy + brake_accuracy) / 3
                
                # Cross-entropy loss (using the distribution's log_prob method)
                cross_entropy_loss = -jnp.mean(actor_dist.log_prob(expert_actions))
                
                # Perplexity
                perplexity = jnp.exp(cross_entropy_loss)
                
                # Action distribution entropy
                action_probs = jax.nn.softmax(actor_dist.logits, axis=-1)
                action_entropy = -jnp.sum(action_probs * jnp.log(action_probs + 1e-8), axis=-1)
                
                # Max probability (confidence)
                max_prob = jnp.max(action_probs, axis=-1)
                
                # Compute action prediction MSE (less meaningful for discrete but still useful)
                action_mse = jnp.mean((predicted_actions - expert_actions) ** 2)
                
                # Compute log probability of actual actions
                log_prob = actor_dist.log_prob(val_batch['actions']).mean()
                
                return {
                    'val_actor_loss': float(actor_loss),
                    'val_action_mse': float(action_mse),
                    'val_categorical_accuracy': float(categorical_accuracy),
                    'val_throttle_accuracy': float(throttle_accuracy),
                    'val_steer_accuracy': float(steer_accuracy),
                    'val_brake_accuracy': float(brake_accuracy),
                    'val_exact_match_accuracy': float(exact_match_accuracy),
                    'val_cross_entropy_loss': float(cross_entropy_loss),
                    'val_perplexity': float(perplexity),
                    'val_action_entropy': float(jnp.mean(action_entropy)),
                    'val_max_probability': float(jnp.mean(max_prob)),
                    'val_log_prob': float(log_prob),
                    'val_bc_log_prob': float(actor_info.get('bc_log_prob', 0.0)),
                    'val_mse': float(actor_info.get('mse', 0.0))
                }
            else:
                # For single discrete actions
                categorical_accuracy = jnp.mean(predicted_actions == expert_actions)
                
                # Cross-entropy loss (using the distribution's log_prob method)
                cross_entropy_loss = -jnp.mean(actor_dist.log_prob(expert_actions))
                
                # Perplexity
                perplexity = jnp.exp(cross_entropy_loss)
                
                # Action distribution entropy
                action_probs = jax.nn.softmax(actor_dist.logits, axis=-1)
                action_entropy = -jnp.sum(action_probs * jnp.log(action_probs + 1e-8), axis=-1)
                
                # Max probability (confidence)
                max_prob = jnp.max(action_probs, axis=-1)
                
                # Compute action prediction MSE (less meaningful for discrete but still useful)
                action_mse = jnp.mean((predicted_actions - expert_actions) ** 2)
        
        # Compute log probability of actual actions
        log_prob = actor_dist.log_prob(val_batch['actions']).mean()
        
        return {
            'val_actor_loss': float(actor_loss),
            'val_action_mse': float(action_mse),
                    'val_categorical_accuracy': float(categorical_accuracy),
                    'val_cross_entropy_loss': float(cross_entropy_loss),
                    'val_perplexity': float(perplexity),
                    'val_action_entropy': float(jnp.mean(action_entropy)),
                    'val_max_probability': float(jnp.mean(max_prob)),
            'val_log_prob': float(log_prob),
            'val_bc_log_prob': float(actor_info.get('bc_log_prob', 0.0)),
            'val_mse': float(actor_info.get('mse', 0.0))
        }
    except Exception as e:
        print(f"Warning: Could not compute validation loss: {e}")
        return {
            'val_actor_loss': float('nan'),
            'val_action_mse': float('nan'),
            'val_categorical_accuracy': float('nan'),
            'val_top_3_accuracy': float('nan'),
            'val_log_prob': float('nan'),
            'val_bc_log_prob': float('nan'),
            'val_mse': float('nan')
        }


def main(args: argparse.Namespace) -> None:
    if args.use_guided:
        agent_module = import_module(f'agents.guided_{args.algorithm.lower()}')
    else:
        agent_module = import_module(f'agents.{args.algorithm.lower()}')
    # Use original algorithm directly
    agent_cls = getattr(agent_module, f'{args.algorithm}Agent')
    get_config = getattr(agent_module, 'get_config')
    
    # Import agent-specific batch sampling function
    if hasattr(agent_module, 'sample_batch'):
        sample_batch_fn = getattr(agent_module, 'sample_batch')
    else:
        # Fallback to CRL sample_batch if agent doesn't provide one
        from agents.crl import sample_batch as crl_sample_batch
        sample_batch_fn = crl_sample_batch
        print(f"Warning: {args.algorithm} doesn't have its own sample_batch function, using CRL's implementation")

    # Get algorithm-specific config
    cfg = get_config()

    # Override with training-specific settings for GCIQL
    cfg.batch_size = args.batch_size
    cfg.actor_loss = 'ddpgbc'  # Use DDPG+BC for stable offline learning
    cfg.expectile = 0.7  # Standard IQL expectile value
    cfg.discount = 0.99  # Reasonable discount for driving task
    cfg.alpha = 0.5  # Moderate BC coefficient for DDPG+BC
    cfg.encoder = 'impala_large'
    cfg.lr = 5e-5  # Much higher learning rate to compensate for small gradients
    cfg.frame_stack = args.frame_stack
    cfg.discrete = True
    cfg.multi_discrete = True  # Enable multi-discrete actor (96 logits, 3D actions)

    dataset = shuffle_dataset(args.dataset_path)
    chunk = load_dataset(dataset, frame_stack=args.frame_stack, chunk_size=args.chunk_size, index=0, config=cfg)

    # Create example action labels for initialization (5 discrete actions)
    ex_action_labels = jnp.zeros(1, dtype=jnp.int32)  # Single example with label 0q

    # Create agent
    agent = agent_cls.create(
        seed=args.seed,
        ex_observations=chunk['observations'][0],  # Use frame-stacked observations
        ex_actions=chunk['actions'][0],
        config=cfg,
    )

    rng = np.random.default_rng(seed=args.seed + 1)

    # Create checkpoint directory
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Progress bar
    total_steps = args.steps * args.epochs
    progress = trange(total_steps, dynamic_ncols=True)

    # Add tracking of actor performance
    reward_stats = []
    actor_losses = []
    bc_losses = []

    # Create validation dataset from last 20% of data
    total_size = len(dataset['observations'])
    val_size = total_size // 5  # Use last 20% for validation
    
    # Create validation dataset by processing the last portion through load_dataset
    val_start_idx = total_size - val_size
    val_dataset = load_dataset(dataset, frame_stack=args.frame_stack, chunk_size=args.chunk_size, index=val_start_idx, config=cfg)
    val_start_idx = val_start_idx + args.chunk_size
    
    print(f"Created validation dataset with {val_size} examples")

    wandb.init(project=args.project, config=cfg.to_dict())
    
    total_steps = 0
    for epoch in range(args.epochs):
        dataset = shuffle_dataset(args.dataset_path)
        chunk = load_dataset(dataset, frame_stack=args.frame_stack, chunk_size=args.chunk_size, index=0, config=cfg)
        print(f"\nStarting epoch {epoch + 1}/{args.epochs}")
        # Progress bar for this epoch
        progress = trange(args.steps, dynamic_ncols=True)
        # Load initial chunk for this epoch
        chunk = load_dataset(dataset, frame_stack=args.frame_stack, chunk_size=args.chunk_size, index=0, config=cfg)
        
        for step in progress:
            # Timing: Start of step
            step_start_time = time.time()
            
            total_steps += 1
            if step % ((args.steps * args.chunk_size) // (dataset['terminals'].shape[0])) == 0 and step < args.steps:
                # Calculate index and ensure it doesn't exceed dataset size
                dataset_size = len(dataset['observations'])
                index = 2 * (step // 5)
                if index >= dataset_size:
                    index = index % dataset_size  # Wrap around
                print(f"Loading chunk at step {step}, epoch {epoch+1}/{args.epochs}, index {index}")
                chunk = load_dataset(dataset, frame_stack=args.frame_stack, chunk_size=args.chunk_size, index=index, config=cfg)
            
            # Timing: Batch sampling
            batch_start_time = time.time()
            batch = sample_batch_fn(
                chunk,
                batch_size=cfg.batch_size,
                frame_stack=args.frame_stack,
                config=cfg
            )
            batch_time = time.time() - batch_start_time

            reward_stats.append(
                {
                    'mean': float(np.mean(batch['rewards'])),
                    'min': float(np.min(batch['rewards'])),
                    'max': float(np.max(batch['rewards'])),
                    'at_goal': float(np.mean(batch['masks'] == 0.0)),
                }
            )

            # Timing: Agent update
            update_start_time = time.time()
            agent, info = agent.update(batch)
            update_time = time.time() - update_start_time
            
            # Timing: Post-processing
            postprocess_start_time = time.time()
            # Track actor loss components
            actor_losses.append(float(info.get('actor/actor_loss', 0.0)))
            bc_losses.append(float(info.get('actor/bc_loss', 0.0)))
            if step % args.log_every == 0:
                # Timing: Logging section
                logging_start_time = time.time()
                log_dict = {k: float(v) for k, v in info.items()}
                
                # Log action predictions
                actor_dist = agent.network.select('actor')(batch['observations'], batch['actor_goals'], params=agent.network.params)
                
                if cfg.discrete:
                    if cfg.get('multi_discrete', False):
                        # For multi-discrete actions (3D), compute accuracy per dimension
                        predicted_actions = actor_dist.mode()
                        expert_actions = batch['actions']
                        
                        throttle_accuracy = jnp.mean(predicted_actions[:, 0] == expert_actions[:, 0])
                        steer_accuracy = jnp.mean(predicted_actions[:, 1] == expert_actions[:, 1])
                        brake_accuracy = jnp.mean(predicted_actions[:, 2] == expert_actions[:, 2])
                        
                        log_dict['actions/throttle_accuracy'] = float(throttle_accuracy)
                        log_dict['actions/steer_accuracy'] = float(steer_accuracy)
                        log_dict['actions/brake_accuracy'] = float(brake_accuracy)
                        log_dict['actions/overall_discrete_accuracy'] = float((throttle_accuracy + steer_accuracy + brake_accuracy) / 3)
                        
                        # Overall exact match accuracy (all dimensions must match)
                        exact_match_accuracy = jnp.mean(jnp.all(predicted_actions == expert_actions, axis=-1))
                        log_dict['actions/exact_match_accuracy'] = float(exact_match_accuracy)
                    else:
                        # For single discrete actions, compute exact match accuracy
                        predicted_actions = actor_dist.mode()
                        expert_actions = batch['actions']
                        categorical_accuracy = jnp.mean(predicted_actions == expert_actions)
                        log_dict['actions/categorical_accuracy'] = float(categorical_accuracy)
                    
                    # Debug: Check for NaN values in key computations
                    log_probs = actor_dist.log_prob(expert_actions)
                    log_dict['debug/log_probs_has_nan'] = float(jnp.any(jnp.isnan(log_probs)))
                    log_dict['debug/logits_has_nan'] = float(jnp.any(jnp.isnan(actor_dist.logits)))
                    
                    # Debug: Check logit ranges and softmax behavior
                    log_dict['debug/logits_min'] = float(jnp.min(actor_dist.logits))
                    log_dict['debug/logits_max'] = float(jnp.max(actor_dist.logits))
                    log_dict['debug/logits_mean'] = float(jnp.mean(actor_dist.logits))
                    log_dict['debug/logits_std'] = float(jnp.std(actor_dist.logits))
                    
                    # Check if logits are in a reasonable range
                    logits_range = jnp.max(actor_dist.logits) - jnp.min(actor_dist.logits)
                    log_dict['debug/logits_range'] = float(logits_range)
                    
                    # Check softmax probabilities
                    action_probs = jax.nn.softmax(actor_dist.logits, axis=-1)
                    log_dict['debug/max_prob'] = float(jnp.max(action_probs))
                    log_dict['debug/min_prob'] = float(jnp.min(action_probs))
                    log_dict['debug/prob_std'] = float(jnp.std(action_probs))
                    
                    log_dict['debug/expert_actions_min'] = float(jnp.min(expert_actions))
                    log_dict['debug/expert_actions_max'] = float(jnp.max(expert_actions))
                    log_dict['debug/expert_actions_unique'] = int(len(jnp.unique(expert_actions)))
                    
                    log_dict['debug/predicted_actions_min'] = float(jnp.min(predicted_actions))
                    log_dict['debug/predicted_actions_max'] = float(jnp.max(predicted_actions))
                    log_dict['debug/predicted_actions_unique'] = int(len(jnp.unique(predicted_actions)))
                    
                    # Cross-entropy loss (using the distribution's log_prob method)
                    cross_entropy_loss = -jnp.mean(log_probs)
                    log_dict['actions/cross_entropy_loss'] = float(cross_entropy_loss)
                    
                    # Perplexity (exponential of cross-entropy, measures model confidence)
                    perplexity = jnp.exp(cross_entropy_loss)
                    log_dict['actions/perplexity'] = float(perplexity)
                    
                    # Action distribution entropy (measures diversity of predictions)
                    action_probs = jax.nn.softmax(actor_dist.logits, axis=-1)
                    action_entropy = -jnp.sum(action_probs * jnp.log(action_probs + 1e-8), axis=-1)
                    log_dict['actions/entropy'] = float(jnp.mean(action_entropy))
                    
                    # Max probability (confidence of the top prediction)
                    max_prob = jnp.max(action_probs, axis=-1)
                    log_dict['actions/max_probability'] = float(jnp.mean(max_prob))
                    
                    # Logit statistics (useful for debugging)
                    logit_mean = jnp.mean(actor_dist.logits, axis=-1)
                    logit_std = jnp.std(actor_dist.logits, axis=-1)
                    log_dict['actions/logit_mean'] = float(jnp.mean(logit_mean))
                    log_dict['actions/logit_std'] = float(jnp.mean(logit_std))
                    
                    # Action space coverage (how many unique actions are being predicted)
                    unique_predicted = len(jnp.unique(predicted_actions))
                    unique_expert = len(jnp.unique(expert_actions))
                    log_dict['actions/unique_predicted'] = unique_predicted
                    log_dict['actions/unique_expert'] = unique_expert
                    log_dict['actions/prediction_diversity'] = unique_predicted / unique_expert if unique_expert > 0 else 0.0
                    
                    # Remove the problematic bincount code for multi-discrete actions
                    # Expert action frequency (how often the expert action appears in the batch)
                    # This doesn't make sense for 3D discrete actions, so we'll skip it
                    # expert_action_freq = jnp.bincount(expert_actions, length=actor_dist.logits.shape[-1])
                    # expert_action_freq = expert_action_freq / jnp.sum(expert_action_freq)
                    # log_dict['actions/expert_action_entropy'] = float(-jnp.sum(expert_action_freq * jnp.log(expert_action_freq + 1e-8)))
                
                # Add actor loss component analysis
                if actor_losses:
                    recent_actor_losses = actor_losses[-args.log_every:]
                    recent_bc_losses = bc_losses[-args.log_every:]
                    
                    log_dict['actor/loss_std'] = float(np.std(recent_actor_losses))
                    log_dict['actor/loss_change'] = (recent_actor_losses[-1] - recent_actor_losses[0]) if len(recent_actor_losses) > 1 else 0
                    log_dict['actor/bc_loss_mean'] = float(np.mean(recent_bc_losses))
                
                wandb.log(log_dict, step=total_steps)

                progress.set_postfix(
                    actor_loss=float(info.get('actor/actor_loss', 0.0)),
                    reward_mean=reward_stats[-1]['mean'] if reward_stats else 0,
                    at_goal=reward_stats[-1]['at_goal'] if reward_stats else 0,
                )
                
                # Timing: End of logging
                logging_time = time.time() - logging_start_time
                if step % 15 == 0:  # Add logging time to our timing report
                    #print(f"Logging time:   {logging_time:.4f}s")
                    pass
            if args.ckpt_every and step and step % args.ckpt_every == 0 or step == 80000:
                ckpt_path = ckpt_dir / f'agent_step{total_steps}.pkl'
                with ckpt_path.open('wb') as f:
                    f.write(fxs.to_bytes(agent))
            # Compute validation loss every 20000 steps
            if step % 10000 == 0 and step > 0:
                print(f"\nComputing validation loss at step {total_steps}...")
                val_metrics = compute_validation_loss(agent, val_dataset, sample_batch_fn, batch_size=1000, frame_stack=args.frame_stack, config=cfg)
                
                print(f"Validation metrics:")
                for key, value in val_metrics.items():
                    print(f"  {key}: {value:.4f}")
                
                # Log validation metrics to wandb
                wandb.log(val_metrics, step=total_steps)
            
            if step % 10000 == 0:
                print(f'Step {total_steps} (Epoch {epoch+1}/{args.epochs}): At-goal rate = {float(jnp.mean(batch["masks"] == 0.0)):.4f}')
    
    final_model_path = ckpt_dir / 'final_model.pkl'
    print(f'Saving final model to {final_model_path}')
    with final_model_path.open('wb') as f:
        f.write(fxs.to_bytes(agent))

    # Final validation evaluation
    print("\nComputing final validation loss...")
    final_val_metrics = compute_validation_loss(agent, val_dataset, sample_batch_fn, batch_size=1000, frame_stack=args.frame_stack, config=cfg)
    
    print("Final validation metrics:")
    for key, value in final_val_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    wandb.log(final_val_metrics, step=step)
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Minimal CRL offline training script')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to .npz offline dataset')
    parser.add_argument('--steps', type=int, default=500_000, help='Total gradient steps')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--actor_loss', choices=['awr', 'ddpgbc'], default='ddpgbc')
    parser.add_argument('--project', default='crl_training')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_every', type=int, default=1000)
    parser.add_argument('--ckpt_every', type=int, default=50_000)
    parser.add_argument('--ckpt_dir', default='checkpoints')
    parser.add_argument('--frame_stack', type=int, default=1)
    parser.add_argument('--algorithm', default='CRL')
    parser.add_argument('--use_guided', action='store_true', help='Use GuidedAgent with CRL as base')
    parser.add_argument('--chunk_size', type=int, default=5000)
    args = parser.parse_args()
    main(args)
