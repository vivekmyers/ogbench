import gymnasium
import numpy as np
from gymnasium.spaces import Box, Discrete
from PIL import Image


class NuPlanEnv(gymnasium.Env):
    """NuPlan environment for offline RL training."""
    
    metadata = {
        'render_modes': None,
    }
    
    def __init__(self, dataset_path=None, config=None):
        """Initialize NuPlan environment.
        
        Args:
            dataset_path: Path to processed .npz dataset
            config: Configuration dict with optional settings:
                - frame_stack: Number of frames to stack (default: None)
                - max_episode_steps: Maximum steps per episode (default: 100)
        """
        self.config = config or {}
        self._max_episode_steps = self.config.get('max_episode_steps', 100)
        self._episode_length = 0
        self._current_idx = None
        
        # Load dataset if path provided
        self.data = None
        if dataset_path is not None:
            # Load processed data
            self.data = dict(np.load(dataset_path))
            
            # Set up observation and action spaces
            obs_shape = self.data['observations'].shape[1:]  # Remove batch dim
            action_shape = self.data['actions'].shape[1:]
            
            # Handle frame stacking if enabled
            if self.config.get('frame_stack'):
                obs_shape = (*obs_shape[:-1], obs_shape[-1] * self.config['frame_stack'])
            
            self.observation_space = Box(
                low=-np.inf,
                high=np.inf,
                shape=obs_shape,
                dtype=np.float32
            )
            
            self.action_space = Box(
                low=-1.0,
                high=1.0,
                shape=action_shape,
                dtype=np.float32
            )
    
    def _get_stacked_obs(self, obs):
        """Stack observations if frame_stack is enabled."""
        frame_stack = self.config.get('frame_stack')
        if frame_stack and frame_stack > 1:
            # For now, just repeat the observation frame_stack times
            # In a real implementation, we'd maintain a buffer of past frames
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
            
        # Sample random episode start
        self._current_idx = np.random.randint(len(self.data['observations']))
        self._episode_length = 0
        
        # Get and process observation
        obs = self.data['observations'][self._current_idx]
        obs = self._get_stacked_obs(obs)
        
        return obs, {}
    
    def step(self, action):
        """Take step in environment using action.
        
        Args:
            action: Action to take (should be in [-1, 1])
            
        Returns:
            observation: Next observation
            reward: Reward (0 for imitation learning)
            terminated: Whether episode ended naturally
            truncated: Whether episode was artificially truncated
            info: Dict with episode information
        """
        if self.data is None:
            raise RuntimeError("Dataset not loaded")
            
        # Get next observation from dataset
        next_idx = (self._current_idx + 1) % len(self.data['observations'])
        next_obs = self.data['next_observations'][self._current_idx]  # Use current idx's next_obs
        next_obs = self._get_stacked_obs(next_obs)
        
        # Get reward from dataset
        reward = float(self.data['rewards'][self._current_idx])
        
        # Update state
        self._current_idx = next_idx
        self._episode_length += 1
        
        # Episode termination conditions:
        # 1. Terminal flag in dataset (end of trajectory)
        # 2. Max episode length reached
        terminated = bool(self.data['terminals'][self._current_idx])
        truncated = self._episode_length >= self._max_episode_steps
        
        # Info dict with episode stats
        info = {
            'episode_length': self._episode_length,
            'success': terminated,  # In NuPlan, terminal usually means success
        }
        
        return next_obs, reward, terminated, truncated, info
    
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


class NuPlanStateEnv(NuPlanEnv):
    """NuPlan environment using state/trajectory data."""
    
    def _process_data(self, raw_data):
        """Process raw data into state representation.
        
        Converts raw data into state vectors containing:
        - Position (x, y)
        - Velocity (vx, vy)
        - Acceleration (ax, ay)
        - Heading (θ)
        - Angular velocity (ω)
        """
        processed = {}
        
        # Extract state vectors from raw data
        # This is a placeholder - we'll implement the actual state extraction
        states = raw_data['observations']  # Shape: (N, state_dim)
        
        processed['observations'] = states.astype(np.float32)
        processed['actions'] = raw_data['actions'].astype(np.float32)
        processed['next_observations'] = raw_data['next_observations'].astype(np.float32)
        processed['terminals'] = raw_data['terminals'].astype(bool)
        processed['value_goals'] = raw_data['value_goals'].astype(np.float32)
        processed['actor_goals'] = raw_data['actor_goals'].astype(np.float32)
        
        return processed
        
    def _make_observation_space(self):
        """Create Box space for state observations."""
        if self.data is None:
            raise RuntimeError("Dataset not loaded")
            
        obs_dim = self.data['observations'].shape[1]
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
    def _make_action_space(self):
        """Create Box space for continuous actions."""
        if self.data is None:
            raise RuntimeError("Dataset not loaded")
            
        action_dim = self.data['actions'].shape[1]
        return Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )


class NuPlanVisualEnv(NuPlanEnv):
    """NuPlan environment using image/video data."""
    
    def _process_data(self, raw_data):
        """Process raw data into visual representation.
        
        Converts raw data into image tensors:
        - RGB images (H, W, 3)
        - Optional depth/segmentation channels
        """
        processed = {}
        
        # Convert to uint8 images and add any additional channels
        # This is a placeholder - we'll implement the actual image processing
        images = raw_data['observations']  # Shape: (N, H, W, C)
        
        processed['observations'] = images.astype(np.uint8)
        processed['actions'] = raw_data['actions'].astype(np.float32)
        processed['next_observations'] = raw_data['next_observations'].astype(np.uint8)
        processed['terminals'] = raw_data['terminals'].astype(bool)
        processed['value_goals'] = raw_data['value_goals'].astype(np.uint8)
        processed['actor_goals'] = raw_data['actor_goals'].astype(np.uint8)
        
        return processed
        
    def _make_observation_space(self):
        """Create Box space for image observations."""
        if self.data is None:
            raise RuntimeError("Dataset not loaded")
            
        obs_shape = self.data['observations'].shape[1:]  # (H, W, C)
        return Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=np.uint8
        )
        
    def _make_action_space(self):
        """Create Box space for continuous actions."""
        if self.data is None:
            raise RuntimeError("Dataset not loaded")
            
        action_dim = self.data['actions'].shape[1]
        return Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )
