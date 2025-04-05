#!/usr/bin/env python
"""
Script to train an agent on the NuPlan environment.
This script integrates the NuPlan environment with the main training workflow.
"""

import os
import sys
import json
import random
import time
from collections import defaultdict

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from agents import agents
from ml_collections import config_flags

# Add the parent directory to the path so we can import from impls
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.env_utils import FrameStackWrapper
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

# Import the NuPlan environment
from ogbench.nuplan import NuplanEnv, NuplanLoader

FLAGS = flags.FLAGS

# Standard flags
flags.DEFINE_string('run_group', 'NuPlan', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

# Training flags
flags.DEFINE_integer('train_steps', 100000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 10000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 50000, 'Saving interval.')

# Evaluation flags
flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all).')
flags.DEFINE_integer('eval_episodes', 10, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('eval_on_cpu', 1, 'Whether to evaluate on CPU.')

# NuPlan-specific flags
flags.DEFINE_string('dataset_path', None, 'Path to the NuPlan dataset.')
flags.DEFINE_string('render_mode', 'video', 'Rendering mode (rgb_array, video, animation).')

config_flags.DEFINE_config_file('agent', 'agents/crl.py', lock_config=False)

def create_nuplan_env_and_datasets(dataset_path, frame_stack=None, config=None):
    """Create NuPlan environment and datasets.
    
    Args:
        dataset_path: Path to the NuPlan dataset.
        frame_stack: Number of frames to stack.
        config: Configuration dictionary.
        
    Returns:
        A tuple of the environment, training dataset, and validation dataset.
    """
    # Create the environment
    env = NuplanEnv(dataset_path=dataset_path, render_mode=FLAGS.render_mode, config=config)
    
    # Create a loader for visualization
    data_dir = os.path.dirname(dataset_path)
    loader = NuplanLoader(data_dir, config)
    
    # Load the dataset
    data = loader.load_dataset(os.path.basename(dataset_path))
    
    # Create training and validation datasets
    # For simplicity, we'll use the same data for both
    # In a real scenario, you would split the data
    train_dataset = {
        'observations': data['observations'],
        'actions': data['actions'],
        'rewards': data['rewards'],
        'next_observations': data['next_observations'],
        'terminals': data['terminals'],
    }
    
    val_dataset = train_dataset.copy()
    
    # Apply frame stacking if requested
    if frame_stack is not None and frame_stack > 1:
        env = FrameStackWrapper(env, frame_stack)
    
    # Reset the environment
    env.reset()
    
    return env, train_dataset, val_dataset

def evaluate_nuplan_agent(agent, env, task_id, num_episodes=5, render=True):
    """Evaluate a NuPlan agent on a specific task.
    
    Args:
        agent: The agent to evaluate.
        env: The NuPlan environment.
        task_id: The task ID to evaluate on.
        num_episodes: Number of episodes to evaluate.
        render: Whether to render the environment.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    # Set up rendering
    if render:
        env.render_mode = FLAGS.render_mode
        env._render_output_path = f'nuplan_task_{task_id}_eval.{FLAGS.render_mode}'
    
    # Evaluate the agent
    metrics = {}
    success_count = 0
    episode_rewards = []
    
    for episode in range(num_episodes):
        # Reset the environment
        obs, info = env.reset(options={'task_id': task_id})
        
        # Run the episode
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            # Get action from agent
            action = agent.sample_actions(obs)
            
            # Step the environment
            obs, reward, done, truncated, info = env.step(action)
            
            # Update metrics
            episode_reward += reward
        
        # Update success count
        if info.get('success', False):
            success_count += 1
        
        # Store episode reward
        episode_rewards.append(episode_reward)
        
        # Render the episode
        if render:
            env.render()
    
    # Calculate metrics
    metrics['success_rate'] = success_count / num_episodes
    metrics['episode_reward'] = np.mean(episode_rewards)
    metrics['episode_reward_std'] = np.std(episode_rewards)
    
    return metrics

def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='OGBench', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Set up environment and dataset.
    config = FLAGS.agent
    
    # Check if dataset path is provided
    if FLAGS.dataset_path is None:
        # Use default path
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'nuplan')
        dataset_path = os.path.join(data_dir, 'nuplan_dataset.npz')
    else:
        dataset_path = FLAGS.dataset_path
    
    # Create environment and datasets
    print(f"Creating NuPlan environment and datasets from {dataset_path}...")
    env, train_dataset, val_dataset = create_nuplan_env_and_datasets(
        dataset_path, 
        frame_stack=config['frame_stack'],
        config=config
    )

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = {
        'observations': train_dataset['observations'][:1],
        'actions': train_dataset['actions'][:1],
    }
    
    if config['discrete']:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Update agent.
        batch_size = min(config['batch_size'], len(train_dataset['observations']))
        indices = np.random.randint(0, len(train_dataset['observations']), size=batch_size)
        
        batch = {
            'observations': train_dataset['observations'][indices],
            'actions': train_dataset['actions'][indices],
            'rewards': train_dataset['rewards'][indices],
            'next_observations': train_dataset['next_observations'][indices],
            'terminals': train_dataset['terminals'][indices],
        }
        
        agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch_size = min(config['batch_size'], len(val_dataset['observations']))
                val_indices = np.random.randint(0, len(val_dataset['observations']), size=val_batch_size)
                
                val_batch = {
                    'observations': val_dataset['observations'][val_indices],
                    'actions': val_dataset['actions'][val_indices],
                    'rewards': val_dataset['rewards'][val_indices],
                    'next_observations': val_dataset['next_observations'][val_indices],
                    'terminals': val_dataset['terminals'][val_indices],
                }
                
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if i == 1 or i % FLAGS.eval_interval == 0:
            if FLAGS.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            else:
                eval_agent = agent
                
            renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            
            # Get task infos
            task_infos = env.task_infos
            num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
            
            for task_id in tqdm.trange(1, num_tasks + 1):
                task_name = task_infos[task_id - 1]['task_name']
                
                # Evaluate agent
                metrics = evaluate_nuplan_agent(
                    eval_agent, 
                    env, 
                    task_id, 
                    num_episodes=FLAGS.eval_episodes, 
                    render=(FLAGS.video_episodes > 0)
                )
                
                # Update metrics
                eval_metrics.update({f'evaluation/{task_name}_{k}': v for k, v in metrics.items()})
                for k, v in metrics.items():
                    overall_metrics[k].append(v)
            
            # Calculate overall metrics
            for k, v in overall_metrics.items():
                eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)
            
            # Log metrics
            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()

if __name__ == '__main__':
    app.run(main) 