from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from ogbench.nuplan.dataset import NuplanDataset


class NuplanEnv(gym.Env):
    """Nuplan environment wrapper.

    This environment wraps the nuplan dataset to create a gymnasium environment.
    The environment uses the offline dataset for observations and actions.
    """

    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 10,
    }

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        render_mode: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the Nuplan environment.

        Args:
            dataset_path: Path to the dataset file
            render_mode: Rendering mode. Either 'human' or 'rgb_array'.
            config: Configuration dictionary for the environment.
        """
        super().__init__()

        self.config = config or {}
        self.frame_stack = self.config.get('frame_stack', 1)
        self._stacked_obs = None
        self._np_random = None

        # Set up observation and action spaces based on dataset inspection
        single_obs_shape = (6,)  # Base observation shape
        if self.frame_stack > 1:
            self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(single_obs_shape[0] * self.frame_stack,), dtype=np.float32
            )
        else:
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=single_obs_shape, dtype=np.float32)

        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(2,),  # Based on dataset inspection
            dtype=np.float32,
        )

        self.render_mode = render_mode

        # Load dataset if path is provided
        self.dataset = None
        if dataset_path is not None:
            self.dataset = NuplanDataset.load(dataset_path, config)

        # Initialize state
        self._current_observation = None
        self._current_step = 0
        self._episode_length = 0
        self._episode_return = 0.0
        self._current_episode_idx = None
        self._episode_data = None

        # For testing: Set arbitrary goals for each episode
        self._current_goal = None
        self._goal_threshold = 0.5  # Distance threshold for reaching goal

    @property
    def task_infos(self):
        """Return task information for evaluation.
        For testing, we'll create three arbitrary tasks with different goals.
        """
        return [
            {'task_name': 'nuplan_short', 'task_id': 1},  # Short-distance navigation
            {'task_name': 'nuplan_medium', 'task_id': 2},  # Medium-distance navigation
            {'task_name': 'nuplan_long', 'task_id': 3},  # Long-distance navigation
        ]

    def _get_stacked_obs(self, obs):
        """Stack observations."""
        if self.frame_stack == 1:
            return obs

        if self._stacked_obs is None:
            self._stacked_obs = np.tile(obs, (self.frame_stack,))
        else:
            self._stacked_obs = np.roll(self._stacked_obs, -obs.shape[0])
            self._stacked_obs[-obs.shape[0] :] = obs

        return self._stacked_obs.copy()

    def _set_goal_for_task(self, task_id=None):
        """Set arbitrary goal based on task ID."""
        if task_id is None or task_id == 1:  # Short distance
            self._current_goal = np.array([2.0, 2.0])
        elif task_id == 2:  # Medium distance
            self._current_goal = np.array([5.0, 5.0])
        else:  # Long distance
            self._current_goal = np.array([10.0, 10.0])

    def _calculate_success(self):
        """Calculate success metric for the current state.
        For testing: Success is reaching within threshold of goal position.
        """
        if self._current_goal is None:
            return False

        # Extract current position from observation (assuming first 2 dimensions are x,y position)
        current_pos = self._current_observation[:2]
        distance_to_goal = np.linalg.norm(current_pos - self._current_goal)

        # Additional test metrics
        collision = False  # In real implementation, check for collisions
        off_road = False  # In real implementation, check if vehicle is off road

        return (distance_to_goal < self._goal_threshold) and not collision and not off_road

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        if seed is not None:
            self._np_random = np.random.RandomState(seed)
        elif self._np_random is None:
            self._np_random = np.random.RandomState()

        if self.dataset is None:
            raise RuntimeError('Dataset not loaded. Please provide dataset_path when creating the environment.')

        # Initialize episode statistics
        self._current_step = 0
        self._episode_length = 0
        self._episode_return = 0.0
        self._stacked_obs = None

        # Get task ID from options
        task_id = options.get('task_id', 1) if options else 1
        self._set_goal_for_task(task_id)

        # Select a random episode using our seeded random state
        self._current_episode_idx = self._np_random.randint(0, self.dataset.num_episodes)
        self._episode_data = self.dataset.get_episode(self._current_episode_idx)

        # Get initial observation
        self._current_observation = self._episode_data['observations'][0]
        stacked_obs = self._get_stacked_obs(self._current_observation)

        info = {
            'episode': {
                'length': 0,
                'return': 0.0,
                'episode_idx': self._current_episode_idx,
            },
            'success': False,
            'goal': self._current_goal,
        }

        return stacked_obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment."""
        if self._episode_data is None:
            raise RuntimeError('Environment not reset. Please call reset() first.')

        # Clip action to action space bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Get next state from dataset
        next_observation = self._episode_data['next_observations'][self._current_step]
        reward = self._episode_data['rewards'][self._current_step]
        terminated = bool(self._episode_data['terminals'][self._current_step])

        # Update state
        self._current_observation = next_observation
        stacked_obs = self._get_stacked_obs(next_observation)
        self._current_step += 1
        self._episode_length += 1
        self._episode_return += reward

        # Check for episode termination
        truncated = self._current_step >= len(self._episode_data['observations'])

        # Calculate success
        success = self._calculate_success()

        info = {
            'episode': {
                'length': self._episode_length,
                'return': self._episode_return,
                'episode_idx': self._current_episode_idx,
            },
            'success': success,
            'goal': self._current_goal,
        }

        return stacked_obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Render the environment.
        For testing: Just show a simple visualization of agent and goal.
        """
        if self.render_mode == 'rgb_array':
            # Create a simple visualization
            frame = np.zeros((100, 100, 3), dtype=np.uint8)

            # Draw current position (red dot)
            pos = self._current_observation[:2]
            x, y = int(pos[0] * 5 + 50), int(pos[1] * 5 + 50)
            x = np.clip(x, 0, 99)
            y = np.clip(y, 0, 99)
            frame[y - 1 : y + 2, x - 1 : x + 2] = [255, 0, 0]

            # Draw goal position (green dot)
            if self._current_goal is not None:
                goal_x = int(self._current_goal[0] * 5 + 50)
                goal_y = int(self._current_goal[1] * 5 + 50)
                goal_x = np.clip(goal_x, 0, 99)
                goal_y = np.clip(goal_y, 0, 99)
                frame[goal_y - 1 : goal_y + 2, goal_x - 1 : goal_x + 2] = [0, 255, 0]

            return frame
        return None

    def close(self) -> None:
        """Close the environment."""
        pass
