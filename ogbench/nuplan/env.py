from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium
from gymnasium.spaces import Box
from ogbench.nuplan.dataset import NuPlanDataset

class NuPlanEnv(gymnasium.Env):
    """Minimal NuPlan environment that provides exactly what main.py needs."""
    
    metadata = {
        'render_modes': ['rgb_array'],
        'render_fps': 30,
    }
    
    def __init__(self, dataset_path=None, render_mode=None, config=None):
        """Initialize environment with dataset and config.
        
        Args:
            dataset_path: Path to .npz dataset file
            render_mode: Rendering mode ('rgb_array' or None)
            config: Optional configuration dict
        """
        super().__init__()
        
        # Store config and initialize random state
        self.config = config or {}
        self.render_mode = render_mode
        self._np_random = None
        
        # Load dataset if path provided
        self.data = None
        if dataset_path is not None:
            # Load dataset using NuPlanDataset
            dataset = NuPlanDataset(dataset_path)
            self.data = dataset.data
            
            # Set up observation and action spaces based on data
            obs_shape = self.data['observations'].shape[1:]  # Remove batch dim
            action_shape = self.data['actions'].shape[1:]
            
            # Handle frame stacking in observation space if enabled
            frame_stack = self.config.get('frame_stack', None)
            if frame_stack is not None and frame_stack > 1:
                obs_shape = (*obs_shape[:-1], obs_shape[-1] * frame_stack)
            
            self.observation_space = Box(
                low=-np.inf, 
                high=np.inf, 
                shape=obs_shape, 
                dtype=np.float32
            )
            self.action_space = Box(
                low=-1.0,  # Normalized action space
                high=1.0,
                shape=action_shape, 
                dtype=np.float32
            )
            
        # Initialize episode state
        self._current_idx = None  # Index in dataset for current episode
        self._current_observation = None
        self._episode_length = 0
        self._max_episode_steps = self.config.get('max_episode_steps', 100)
        
    def _get_stacked_obs(self, obs):
        """Stack observations if frame_stack is enabled.
        
        Args:
            obs: Single observation to potentially stack
            
        Returns:
            Original or stacked observation depending on config
        """
        frame_stack = self.config.get('frame_stack', None)
        if frame_stack is not None and frame_stack > 1:
            # Repeat the observation frame_stack times along last axis
            return np.tile(obs, frame_stack)
        return obs
        
    def reset(self, seed=None, options=None):
        """Reset environment to start new episode.
        
        Args:
            seed: Random seed (unused)
            options: Optional dict with settings (unused)
            
        Returns:
            observation: Initial observation
            info: Empty dict
        """
        if self.data is None:
            raise RuntimeError("Dataset not loaded")
            
        # Sample random episode start from dataset
        self._current_idx = np.random.randint(len(self.data['observations']))
        self._current_observation = self._get_stacked_obs(
            self.data['observations'][self._current_idx]
        )
        self._episode_length = 0
        
        return self._current_observation, {}
        
    def step(self, action):
        """Take step in environment using action.
        
        Args:
            action: Action to take (should be in [-1, 1])
            
        Returns:
            observation: Next observation
            reward: Reward from dataset
            terminated: Whether episode ended naturally 
            truncated: Whether episode was artificially truncated
            info: Dict with episode information
        """
        if self.data is None:
            raise RuntimeError("Dataset not loaded")
            
        # Get next observation and reward from dataset
        next_idx = (self._current_idx + 1) % len(self.data['observations'])
        next_observation = self._get_stacked_obs(
            self.data['observations'][next_idx]
        )
        reward = float(self.data['rewards'][self._current_idx])
        
        # Update state
        self._current_idx = next_idx
        self._current_observation = next_observation
        self._episode_length += 1
        
        # Episode termination conditions
        terminated = bool(self.data['terminals'][self._current_idx])
        truncated = self._episode_length >= self._max_episode_steps
        
        # Info dict with episode stats
        info = {
            'episode_length': self._episode_length,
            'success': terminated,  # In NuPlan, terminal usually means success
        }
        
        return next_observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the current environment state.
        
        Returns:
            If render_mode is 'rgb_array', returns a numpy array of the rendered frame.
            Otherwise returns None.
        """
        if self.render_mode == 'rgb_array':
            # For now, just return a blank frame since we don't have visualization yet
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return None
    
    @property 
    def task_infos(self):
        """Return task information required by main.py's evaluation.
        
        Returns:
            List containing single task info dict
        """
        return [{
            'task_name': 'nuplan',
            'task_id': 1,
            'max_episode_steps': self._max_episode_steps,
        }]
