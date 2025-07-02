from __future__ import annotations

import argparse
import time
from importlib import import_module
from pathlib import Path
from typing import Dict

import flax.serialization as fxs
import jax.numpy as jnp
import numpy as np
import wandb
from tqdm import trange


def load_dataset(path: str | Path, frame_stack: int = 1) -> Dict[str, jnp.ndarray]:
    start = time.time()
    data_np = np.load(path)
    dataset = {k: np.asarray(v, dtype=np.float32) for k, v in data_np.items()}

    print(f'Dataset loaded: {len(dataset["observations"])} transitions')
    print(f'Terminal states: {np.sum(dataset["terminals"])}')
    print(f'Observation shape: {dataset["observations"].shape}, dtype: {dataset["observations"].dtype}')

    print(dataset['observations'].shape, dataset['observations'].size)
    # dataset['observations'] = dataset['observations'][:50000]
    # dataset['terminals'] = dataset['terminals'][:50000]

    print('Converting to jnp')
    dataset = {k: jnp.array(v) for k, v in dataset.items()}

    original_obs = dataset['observations'].reshape(45999, 48, 48, 3).astype(np.float32)
    print(original_obs.shape)

    n_transitions, h, w, c = 45999, 48, 48, 3

    terminals = dataset['terminals'][:-3]
    # terminals = terminals.at[-1].set(True)
    terminals = terminals.at[-1].set(True)
    dataset['terminals'] = terminals

    shifted_dataset = [jnp.roll(original_obs, shift=i, axis=0) for i in range(frame_stack - 1, -1, -1)]
    concat_dataset = jnp.concatenate(shifted_dataset, axis=-1)
    concat_dataset = concat_dataset[(frame_stack - 1) :]
    dataset['observations'] = concat_dataset
    next_obs = jnp.roll(concat_dataset, shift=-1, axis=0)
    # next_obs = next_obs.at[-1].set(next_obs[-2])
    next_obs = next_obs.at[-1].set(next_obs[-2])
    dataset['next_observations'] = next_obs

    end = time.time()
    print(f'loading step complete: {end - start:.3f} seconds')
    print(f'observations shape: {dataset["observations"].shape}')
    print(f'next_observations shape: {dataset["next_observations"].shape}')
    print(f'terminals shape: {dataset["terminals"].shape}')
    return dataset


def sample_batch(
    dataset: Dict[str, jnp.ndarray], *, batch_size: int, frame_stack: int = 1, config: Dict
) -> Dict[str, jnp.ndarray]:
    n = dataset['observations'].shape[0]
    idx = np.random.choice(n, batch_size, replace=True)

    batch = {
        'observations': dataset['observations'][idx],
        'actions': dataset['actions'][idx],
        'next_observations': dataset['next_observations'][idx],
        'rewards': jnp.zeros((batch_size,)),
        'masks': jnp.ones((batch_size,)),
    }

    # use_future = np.random.uniform(size=(batch_size,)) < 0.25

    # traj_ends = jnp.searchsorted(terminal_indices, idx, side='right')
    # Sample from further in the future (e.g., 5-20 steps ahead) instead of just idx + 1
    # future_steps = np.random.randint(5, 21, size=(batch_size,))  # Random future steps between 5-20
    # middle_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
    # future_idx = jnp.minimum(idx + future_steps, terminal_indices[jnp.minimum(traj_ends, len(terminal_indices) - 1)])

    # Random goals
    # random_idx = np.random.choice(n, batch_size, replace=True)

    # goal_idx = jnp.where(use_future, future_idx, random_idx)

    offsets = np.random.geometric(p=1 - config['discount'], size=batch_size)  # in [1, inf)
    goal_idx = np.clip(idx + offsets, 0, n - 1)
    base_goals = dataset['observations'][goal_idx][:, :, :, -3:]
    goal_stack = jnp.concatenate([base_goals] * frame_stack, axis=-1)

    batch['value_goals'] = goal_stack
    batch['actor_goals'] = goal_stack

    batch['rewards'] = -1.0 * jnp.ones((batch_size,))
    batch['masks'] = jnp.ones((batch_size,))

    next_obs_flat = batch['next_observations'].reshape(batch_size, -1)
    goals_flat = batch['value_goals'].reshape(batch_size, -1)

    distances = jnp.sqrt(np.sum((next_obs_flat - goals_flat) ** 2, axis=1)) / jnp.sqrt(next_obs_flat.shape[1])

    # Use a more reasonable goal threshold based on distance percentiles
    target_goal_rate = 0.01
    sorted_distances = jnp.sort(distances)
    percentile_idx = int(target_goal_rate * batch_size)
    goal_threshold = (
        sorted_distances[percentile_idx] if percentile_idx < len(sorted_distances) else sorted_distances[-1]
    )
    is_at_goal = distances <= goal_threshold

    epsilon = 1e-6
    normalized_distances = distances / (goal_threshold + epsilon)
    smooth_rewards = jnp.clip(-normalized_distances, -1.0, 0.0)
    batch['rewards'] = smooth_rewards
    batch['masks'] = jnp.where(is_at_goal, 0.0, 1.0)  # Set masks to 0 when at goal

    return batch


def main(args: argparse.Namespace) -> None:
    # Import agent module and get config first
    agent_module = import_module(f'agents.{args.algorithm.lower()}')
    agent_cls = getattr(agent_module, f'{args.algorithm}Agent')
    get_config = getattr(agent_module, 'get_config')

    # Get algorithm-specific config
    cfg = get_config()

    # Override with training-specific settings
    cfg.batch_size = args.batch_size
    cfg.actor_loss = 'ddpgbc'
    cfg.encoder = 'impala'
    cfg.value_p_curgoal = 0.0
    cfg.value_p_trajgoal = 0.5
    cfg.value_p_randomgoal = 0.5
    cfg.actor_p_curgoal = 0.0
    cfg.actor_p_trajgoal = 0.5
    cfg.actor_p_rangomdoal = 0.5
    cfg.value_geom_sample = True
    cfg.actor_geom_sample = True
    cfg.gc_negative = True
    cfg.temperature = 0.2
    cfg.lr = 1e-4
    cfg.replay_size = 1000000
    cfg.target_update_interval = 1
    cfg.critic_train_freq = 1
    cfg.latent_dim = 256
    cfg.discount = 0.95

    wandb.init(project=args.project, config=cfg.to_dict())
    dataset = load_dataset(args.dataset_path, frame_stack=args.frame_stack)

    agent = agent_cls.create(
        seed=args.seed,
        ex_observations=dataset['observations'][0],
        ex_actions=dataset['actions'][0],
        config=cfg,
    )

    rng = np.random.default_rng(seed=args.seed + 1)

    # Create checkpoint directory
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Progress bar
    progress = trange(args.steps, dynamic_ncols=True)

    # Add tracking of critic performance
    critic_losses = []
    reward_stats = []

    for step in progress:
        batch = sample_batch(
            dataset,
            batch_size=cfg.batch_size,
            frame_stack=args.frame_stack,
            config=cfg,
        )

        if step % 2000 == 100:
            print('Masks:', np.unique(batch['masks'], return_counts=True))

        reward_stats.append(
            {
                'mean': float(np.mean(batch['rewards'])),
                'min': float(np.min(batch['rewards'])),
                'max': float(np.max(batch['rewards'])),
                'at_goal': float(np.mean(batch['masks'] == 0.0)),
            }
        )

        if step % 10000 == 0:
            print(
                f'Step {step}: at_goal_rate = {float(np.mean(batch["masks"] == 0.0)):.4f}, '
                f'reward_mean = {float(np.mean(batch["rewards"])):.4f}, '
                f'reward_std = {float(np.std(batch["rewards"])):.4f}, '
                f'reward_min = {float(np.min(batch["rewards"])):.4f}, '
                f'reward_max = {float(np.max(batch["rewards"])):.4f}'
            )
            print("DEBUG: batch['observations'] shape:", batch['observations'].shape)
            print("DEBUG: batch['next_observations'] shape:", batch['next_observations'].shape)
            print("DEBUG: batch['value_goals'] shape:", batch['value_goals'].shape)
            print("DEBUG: batch['actor_goals'] shape:", batch['actor_goals'].shape)
            print('DEBUG: Example obs[0]:', np.array(batch['observations'][0]))
            print('DEBUG: Example next_obs[0]:', np.array(batch['next_observations'][0]))
            print('DEBUG: Example value_goal[0]:', np.array(batch['value_goals'][0]))

        agent, info = agent.update(batch)

        critic_losses.append(float(info.get('critic/contrastive_loss', 0.0)))

        if step % args.log_every == 0:
            log_dict = {k: float(v) for k, v in info.items()}
            if reward_stats:
                recent_rewards = reward_stats[-args.log_every :]
                log_dict.update(
                    {
                        'rewards/mean': float(np.mean([s['mean'] for s in recent_rewards])),
                        'rewards/min': float(np.min([s['min'] for s in recent_rewards])),
                        'rewards/max': float(np.max([s['max'] for s in recent_rewards])),
                        'rewards/at_goal_rate': float(np.mean([s['at_goal'] for s in recent_rewards])),
                    }
                )

            if critic_losses:
                recent_losses = critic_losses[-args.log_every :]
                log_dict['critic/loss_std'] = float(np.std(recent_losses))
                log_dict['critic/loss_change'] = (recent_losses[-1] - recent_losses[0]) if len(recent_losses) > 1 else 0
            wandb.log(log_dict, step=step)

            progress.set_postfix(
                critic_loss=float(info.get('critic/contrastive_loss', 0.0)),
                reward_mean=reward_stats[-1]['mean'] if reward_stats else 0,
                at_goal=reward_stats[-1]['at_goal'] if reward_stats else 0,
            )
        if args.ckpt_every and step and step % args.ckpt_every == 0:
            ckpt_path = ckpt_dir / f'agent_step{step}.pkl'
            with ckpt_path.open('wb') as f:
                f.write(fxs.to_bytes(agent))
        if step % 10000 == 0:
            print(f'Step {step}: At-goal rate = {float(jnp.mean(batch["masks"] == 0.0)):.4f}')
    final_model_path = ckpt_dir / 'final_model.pkl'
    print(f'Saving final model to {final_model_path}')
    with final_model_path.open('wb') as f:
        f.write(fxs.to_bytes(agent))
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Minimal CRL offline training script')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to .npz offline dataset')
    parser.add_argument('--steps', type=int, default=700_000, help='Total gradient steps')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--actor_loss', choices=['awr', 'ddpgbc'], default='ddpgbc')
    parser.add_argument('--project', default='crl_training')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_every', type=int, default=1000)
    parser.add_argument('--ckpt_every', type=int, default=50_000)
    parser.add_argument('--ckpt_dir', default='checkpoints')
    parser.add_argument('--frame_stack', type=int, default=1)
    parser.add_argument('--algorithm', default='CRL')
    args = parser.parse_args()
    main(args)
