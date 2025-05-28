#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path
import os
import sys
import time
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import flax.serialization as fxs
import gymnasium as gym

# Import the CRL agent
from agents.crl import CRLAgent

# Add parent directory to path for ogbench imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Now import CarlaEnv
from ogbench.CarlaEnv import CarlaEnv


class FrameStack:
    """Maintains a frame stack of observations for evaluation."""
    
    def __init__(self, num_frames, obs_shape):
        """Initialize a frame stack buffer.
        
        Args:
            num_frames: Number of frames to stack
            obs_shape: Shape of a single observation
        """
        self.num_frames = num_frames
        self.frames = None
        self.obs_shape = obs_shape
        
    def reset(self, first_obs):
        """Reset the frame stack with the first observation.
        
        Args:
            first_obs: First observation to fill the stack with
        """
        # Normalize observation if it's an image
        if len(first_obs.shape) == 3 and first_obs.shape[-1] in [1, 3]:
            first_obs = first_obs / 255.0
            
        # For images
        if len(first_obs.shape) == 3:
            h, w, c = first_obs.shape
            # Initialize all frames to be the first observation
            self.frames = np.zeros((h, w, c * self.num_frames), dtype=np.float32)
            
            # Fill all frames with the first observation
            for i in range(self.num_frames):
                self.frames[:, :, i * c:(i + 1) * c] = first_obs
        else:
            # For vector observations
            self.frames = np.tile(first_obs, self.num_frames)
        
        return self.frames.copy()
        
    def update(self, new_obs):
        """Add a new observation to the stack and return the updated stack.
        
        Args:
            new_obs: New observation to add to the stack
        
        Returns:
            Updated frame stack
        """
        # Normalize observation if it's an image
        if len(new_obs.shape) == 3 and new_obs.shape[-1] in [1, 3]:
            new_obs = new_obs / 255.0
            
        # For images
        if len(new_obs.shape) == 3:
            h, w, c = new_obs.shape
            
            # Shift frames and add new one
            self.frames[:, :, :-c] = self.frames[:, :, c:]
            self.frames[:, :, -c:] = new_obs
        else:
            # For vector observations
            d = new_obs.shape[0]
            self.frames[:-d] = self.frames[d:]
            self.frames[-d:] = new_obs
        
        return self.frames.copy()


def evaluate_policy(agent, env, num_episodes=10, render=False, frame_stack=1):
    """Evaluate a policy on an environment.
    
    Args:
        agent: The agent to evaluate
        env: The environment to evaluate on
        num_episodes: Number of episodes to run
        render: Whether to render the environment
        frame_stack: Number of frames to stack
    
    Returns:
        Dictionary of evaluation metrics
    """
    rewards = []
    collisions = []
    successes = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        had_collision = False
        
        # Setup frame stacking
        frame_stacker = FrameStack(frame_stack, obs.shape)
        stacked_obs = frame_stacker.reset(obs)
        
        while not done:
            if render:
                env.render()
                time.sleep(0.05)
            
            # Normalize observations
            # CARLA provides RGB images in range 0-255
            norm_obs = stacked_obs
            
            # For goal-conditioned policies, we need a goal
            # In CARLA, the goal could be the target destination or a future waypoint
            goal = norm_obs  # For now, use the current observation as the goal
            
            # Sample action from policy
            # Note: CRL agent expects batch dimensions
            obs_batch = jnp.expand_dims(jnp.asarray(norm_obs), 0)
            goal_batch = jnp.expand_dims(jnp.asarray(goal), 0)
            
            # Action space is normalized to [-1, 1]
            action = agent.sample_actions(obs_batch, goal_batch, argmax=True)[0]
            
            # Execute action
            next_obs, reward, done, info = env.step(action)
            
            # Update frame stack
            stacked_obs = frame_stacker.update(next_obs)
            
            episode_reward += reward
            
            # Track collisions and successes
            if hasattr(env, 'collision_detected') and env.collision_detected:
                had_collision = True
            
            # In CARLA, success might be reaching a destination
            # This would be in the info dict
            success = info.get('success', False)
            
            if done:
                successes.append(success)
                collisions.append(had_collision)
                
        rewards.append(episode_reward)
        print(f"Episode {episode}: Reward = {episode_reward}, Collision = {had_collision}, Success = {success}")
    
    # Compute metrics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    collision_rate = np.mean(collisions)
    success_rate = np.mean(successes)
    
    # Return metrics
    metrics = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "min_reward": min_reward,
        "max_reward": max_reward,
        "collision_rate": collision_rate,
        "success_rate": success_rate,
    }
    
    print("\nEvaluation Results:")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Min/Max Reward: {min_reward:.2f}/{max_reward:.2f}")
    print(f"Collision Rate: {collision_rate:.2f}")
    print(f"Success Rate: {success_rate:.2f}")
    
    return metrics


def main(args):
    # Load agent from checkpoint
    print(f"Loading agent from {args.checkpoint}")
    with open(args.checkpoint, "rb") as f:
        agent = fxs.from_bytes(CRLAgent, f.read())
    
    # Create environment directly using CarlaEnv (not through gym.make)
    print(f"Creating CARLA environment")
    
    # Prepare arguments for CarlaEnv
    carla_args = {
        "vision_size": args.vision_size,
        "vision_fov": args.vision_fov,
        "weather": args.weather,
        "frame_skip": args.frame_skip,
        "multiagent": True,
        "lane": 0,
        "lights": True,
        "steps": args.steps,
    }
    
    env = CarlaEnv(
        render=args.render,
        carla_port=args.port, 
        reward_type=args.reward_type,
        frame_skip=args.frame_skip,
        vision_size=args.vision_size,
        vision_fov=args.vision_fov,
        args=carla_args
    )
    
    # Print environment info
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Run evaluation
    print(f"Starting evaluation for {args.num_episodes} episodes...")
    metrics = evaluate_policy(
        agent, 
        env, 
        num_episodes=args.num_episodes, 
        render=args.render,
        frame_stack=args.frame_stack
    )
    
    # Print summary
    print("\nEvaluation Results:")
    print(f"Mean reward: {metrics['mean_reward']:.2f}")
    print(f"Std reward: {metrics['std_reward']:.2f}")
    print(f"Min reward: {metrics['min_reward']:.2f}")
    print(f"Max reward: {metrics['max_reward']:.2f}")
    print(f"Collision rate: {metrics['collision_rate']*100:.1f}%")
    print(f"Success rate: {metrics['success_rate']*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained policy in CARLA")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to agent checkpoint")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--port", type=int, default=2000, help="CARLA simulator port")
    parser.add_argument("--reward_type", type=str, default="goal_reaching", choices=["goal_reaching", "lane_follow"], help="Reward type")
    parser.add_argument("--frame_skip", type=int, default=4, help="Frame skip")
    parser.add_argument("--vision_size", type=int, default=48, help="Vision observation size")
    parser.add_argument("--vision_fov", type=int, default=90, help="Field of view for camera")
    parser.add_argument("--weather", type=float, default=0.0, help="Weather condition (0.0 for clear)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--frame_stack", type=int, default=1, help="Number of frames to stack (default: 1, no stacking)")
    parser.add_argument("--steps", type=int, default=1000, help="Max steps per episode")
    
    args = parser.parse_args()
    main(args) 