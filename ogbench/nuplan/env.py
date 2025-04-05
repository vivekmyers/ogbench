from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium.spaces import Box

class NuPlanEnv:
    """Minimal NuPlan environment that provides exactly what main.py needs."""
    
    def __init__(self, dataset_path=None, config=None):
        """Initialize environment with dataset and config.
        
        Args:
            dataset_path: Path to .npz dataset file
            config: Optional configuration dict
        """
        # Store config and initialize random state
        self.config = config or {}
        self._np_random = None
        
        # Load dataset if path provided
        self.data = None
        if dataset_path is not None:
            # Load raw data
            raw_data = dict(np.load(dataset_path))
            
            # Add goal fields as copies of next_observations
            raw_data['value_goals'] = raw_data['next_observations'].copy()
            raw_data['actor_goals'] = raw_data['next_observations'].copy()
            
            # Store processed data
            self.data = {k: v.astype(np.float32) for k, v in raw_data.items()}
            
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
                low=-np.inf, 
                high=np.inf, 
                shape=action_shape, 
                dtype=np.float32
            )
            
        # Initialize episode state
        self._current_idx = None  # Index in dataset for current episode
        self._current_observation = None
        self._episode_length = 0
        
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
        
    def reset(self):
        """Reset environment to start new episode.
        
        Returns:
            observation: Initial observation
            info: Empty dict (no special info needed)
        """
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
            action: Action to take
            
        Returns:
            observation: Next observation
            reward: Reward (0 for now since we're just doing imitation)
            terminated: Whether episode ended naturally 
            truncated: Whether episode was artificially truncated
            info: Empty dict (no special info needed)
        """
        # Get next observation from dataset
        next_idx = (self._current_idx + 1) % len(self.data['observations'])
        next_observation = self._get_stacked_obs(
            self.data['observations'][next_idx]
        )
        
        # Update state
        self._current_idx = next_idx
        self._current_observation = next_observation
        self._episode_length += 1
        
        # Simple episode termination after 100 steps
        terminated = self.data['terminals'][self._current_idx]
        truncated = self._episode_length >= 100
        
        return next_observation, 0.0, terminated, truncated, {}
    
    @property 
    def task_infos(self):
        """Return task information required by main.py's evaluation.
        
        Returns:
            List containing single task info dict
        """
        return [{'task_name': 'nuplan', 'task_id': 1}]
