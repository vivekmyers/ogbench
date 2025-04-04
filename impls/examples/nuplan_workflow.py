#!/usr/bin/env python
"""
Example script demonstrating how to connect the NuPlan environment to the existing workflow
and set up video rendering.
"""

import os
import sys
import numpy as np
import jax
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

# Add the parent directory to the path so we can import from impls
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.env_utils import make_env_and_datasets, FrameStackWrapper
from utils.evaluation import evaluate
from agents import agents
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, setup_wandb

# Import the NuPlan environment
from ogbench.nuplan import NuplanEnv, NuplanLoader

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
    env = NuplanEnv(dataset_path=dataset_path, render_mode='video', config=config)
    
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
        env.render_mode = 'video'
        env._render_output_path = f'nuplan_task_{task_id}_eval.mp4'
    
    # Evaluate the agent
    metrics = {}
    success_count = 0
    
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
        
        # Render the episode
        if render:
            env.render()
    
    # Calculate metrics
    metrics['success_rate'] = success_count / num_episodes
    metrics['episode_reward'] = episode_reward / num_episodes
    
    return metrics

def main():
    # Set up paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'nuplan')
    dataset_path = os.path.join(data_dir, 'nuplan_dataset.npz')
    
    # Create output directory for videos
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up configuration
    config = {
        'frame_stack': 1,
        'action_repeat': 1,
        'seed': 42,
        'batch_size': 256,
        'agent_name': 'crl',
        'learning_rate': 3e-4,
    }
    
    # Create environment and datasets
    print("Creating NuPlan environment and datasets...")
    env, train_dataset, val_dataset = create_nuplan_env_and_datasets(
        dataset_path, 
        frame_stack=config['frame_stack'],
        config=config
    )
    
    # Initialize agent
    print("Initializing agent...")
    example_batch = {
        'observations': train_dataset['observations'][:1],
        'actions': train_dataset['actions'][:1],
    }
    
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        config['seed'],
        example_batch['observations'],
        example_batch['actions'],
        config,
    )
    
    # Train agent (simplified for demonstration)
    print("Training agent...")
    for i in tqdm(range(100)):
        # Sample batch
        batch_size = min(config['batch_size'], len(train_dataset['observations']))
        indices = np.random.randint(0, len(train_dataset['observations']), size=batch_size)
        
        batch = {
            'observations': train_dataset['observations'][indices],
            'actions': train_dataset['actions'][indices],
            'rewards': train_dataset['rewards'][indices],
            'next_observations': train_dataset['next_observations'][indices],
            'terminals': train_dataset['terminals'][indices],
        }
        
        # Update agent
        agent, update_info = agent.update(batch)
        
        # Print progress
        if i % 10 == 0:
            print(f"Step {i}: {update_info}")
    
    # Evaluate agent
    print("Evaluating agent...")
    for task_id in range(1, 4):  # Evaluate on all three tasks
        metrics = evaluate_nuplan_agent(
            agent, 
            env, 
            task_id, 
            num_episodes=3, 
            render=True
        )
        
        print(f"Task {task_id} metrics: {metrics}")
    
    # Save agent
    print("Saving agent...")
    save_agent(agent, output_dir, 100)
    
    print("Done!")

if __name__ == "__main__":
    main() 