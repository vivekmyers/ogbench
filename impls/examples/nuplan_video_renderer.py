#!/usr/bin/env python
"""
Script for rendering videos with the NuPlan environment.
This script demonstrates different ways to render videos from the NuPlan environment.
"""

import os
import sys
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# Add the parent directory to the path so we can import from impls
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the NuPlan environment
from ogbench.nuplan import NuplanEnv, NuplanLoader

def render_episode(env, task_id, output_path, render_mode='video'):
    """Render an episode from the NuPlan environment.
    
    Args:
        env: The NuPlan environment.
        task_id: The task ID to render.
        output_path: Path to save the rendered video.
        render_mode: Rendering mode ('video' or 'animation').
    """
    # Set up rendering
    env.render_mode = render_mode
    env._render_output_path = output_path
    
    # Reset the environment
    obs, info = env.reset(options={'task_id': task_id})
    
    # Run the episode
    done = False
    truncated = False
    
    while not (done or truncated):
        # Sample random action
        action = env.action_space.sample()
        
        # Step the environment
        obs, reward, done, truncated, info = env.step(action)
        
        # Render the environment
        env.render()
    
    print(f"Rendered episode saved to {output_path}")

def render_multiple_episodes(env, task_id, num_episodes, output_dir, render_mode='video'):
    """Render multiple episodes from the NuPlan environment.
    
    Args:
        env: The NuPlan environment.
        task_id: The task ID to render.
        num_episodes: Number of episodes to render.
        output_dir: Directory to save the rendered videos.
        render_mode: Rendering mode ('video' or 'animation').
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Render episodes
    for i in range(num_episodes):
        output_path = os.path.join(output_dir, f'nuplan_task_{task_id}_episode_{i}.{render_mode}')
        render_episode(env, task_id, output_path, render_mode)
    
    print(f"Rendered {num_episodes} episodes saved to {output_dir}")

def render_from_dataset(loader, dataset_path, output_path, num_frames=50, render_mode='video'):
    """Render a video from a dataset.
    
    Args:
        loader: The NuPlan loader.
        dataset_path: Path to the dataset.
        output_path: Path to save the rendered video.
        num_frames: Number of frames to render.
        render_mode: Rendering mode ('video' or 'animation').
    """
    # Load dataset
    data = loader.load_dataset(dataset_path)
    
    # Extract observations and actions
    observations = data['observations'][:num_frames]
    actions = data['actions'][:num_frames]
    
    # Create goal (example)
    goal = np.array([5.0, 5.0])
    
    # Render video
    if render_mode == 'video':
        loader.create_video(observations, actions, goal, output_path)
    else:
        loader.create_animation(observations, actions, goal, output_path)
    
    print(f"Rendered video saved to {output_path}")

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
    }
    
    # Create environment
    print("Creating NuPlan environment...")
    env = NuplanEnv(dataset_path=dataset_path, render_mode='video', config=config)
    
    # Create loader
    print("Creating NuPlan loader...")
    loader = NuplanLoader(data_dir, config)
    
    # Example 1: Render a single episode
    print("\nExample 1: Rendering a single episode")
    render_episode(
        env, 
        task_id=1, 
        output_path=os.path.join(output_dir, 'nuplan_task_1_episode.mp4'),
        render_mode='video'
    )
    
    # Example 2: Render multiple episodes
    print("\nExample 2: Rendering multiple episodes")
    render_multiple_episodes(
        env, 
        task_id=2, 
        num_episodes=3, 
        output_dir=os.path.join(output_dir, 'multiple_episodes'),
        render_mode='video'
    )
    
    # Example 3: Render from dataset
    print("\nExample 3: Rendering from dataset")
    try:
        render_from_dataset(
            loader, 
            os.path.basename(dataset_path), 
            os.path.join(output_dir, 'nuplan_dataset.mp4'),
            num_frames=50,
            render_mode='video'
        )
    except FileNotFoundError:
        print(f"Dataset not found at {dataset_path}. Please provide a valid dataset path.")
    
    # Example 4: Render animation
    print("\nExample 4: Rendering animation")
    render_episode(
        env, 
        task_id=3, 
        output_path=os.path.join(output_dir, 'nuplan_task_3_episode.gif'),
        render_mode='animation'
    )
    
    print("\nDone!")

if __name__ == "__main__":
    main() 