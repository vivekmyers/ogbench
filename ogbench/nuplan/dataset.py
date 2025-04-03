from typing import Any, Dict, Optional

import numpy as np


class NuplanDataset:
    """Handler for nuplan dataset.

    This class manages loading and accessing the nuplan dataset.
    """

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the dataset.

        Args:
            data: Dictionary containing the dataset arrays
            config: Configuration dictionary
        """
        self.data = data
        self.config = config or {}

        # Validate required keys
        required_keys = ['observations', 'actions', 'terminals']
        for key in required_keys:
            if key not in self.data:
                raise ValueError(f'Missing required key in dataset: {key}')

        # Process data
        self._process_data()

        # Initialize episode boundaries
        self._init_episode_boundaries()

    def _process_data(self) -> None:
        """Process the dataset arrays."""
        # Ensure arrays are float32
        self.data['observations'] = self.data['observations'].astype(np.float32)
        self.data['actions'] = self.data['actions'].astype(np.float32)

        # Add rewards if not present
        if 'rewards' not in self.data:
            self.data['rewards'] = np.zeros(len(self.data['observations']), dtype=np.float32)

        # Compute next observations
        self.data['next_observations'] = np.roll(self.data['observations'], -1, axis=0)

        # Handle terminal states
        terminal_mask = self.data['terminals'].astype(bool)
        self.data['next_observations'][terminal_mask] = 0.0

    def _init_episode_boundaries(self) -> None:
        """Initialize episode boundaries."""
        # Find indices where episodes start
        self.episode_starts = np.where(self.data['terminals'])[0] + 1
        self.episode_starts = np.concatenate([[0], self.episode_starts])

        # Find indices where episodes end
        self.episode_ends = np.where(self.data['terminals'])[0]

        # Number of episodes
        self.num_episodes = len(self.episode_starts)

    @classmethod
    def load(
        cls,
        path: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> 'NuplanDataset':
        """Load dataset from file.

        Args:
            path: Path to the dataset file
            config: Configuration dictionary

        Returns:
            NuplanDataset instance
        """
        data = np.load(path)
        return cls(dict(data), config)

    def save(self, path: str) -> None:
        """Save dataset to file.

        Args:
            path: Path to save the dataset
        """
        np.savez(path, **self.data)

    def get_subset(self, indices: np.ndarray) -> 'NuplanDataset':
        """Get a subset of the dataset.

        Args:
            indices: Indices to select

        Returns:
            New NuplanDataset instance with selected data
        """
        subset_data = {k: v[indices] for k, v in self.data.items()}
        return NuplanDataset(subset_data, self.config)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary containing the sampled batch
        """
        indices = np.random.randint(0, len(self.data['observations']), size=batch_size)
        return {
            'observations': self.data['observations'][indices],
            'actions': self.data['actions'][indices],
            'rewards': self.data['rewards'][indices],
            'next_observations': self.data['next_observations'][indices],
            'terminals': self.data['terminals'][indices],
        }

    def get_episode(self, episode_idx: int) -> Dict[str, np.ndarray]:
        """Get a complete episode.

        Args:
            episode_idx: Index of the episode to get

        Returns:
            Dictionary containing the episode data
        """
        start = self.episode_starts[episode_idx]
        end = self.episode_ends[episode_idx] + 1

        # Get episode data
        episode_data = {
            'observations': self.data['observations'][start:end],
            'actions': self.data['actions'][start:end],
            'rewards': self.data['rewards'][start:end],
            'terminals': self.data['terminals'][start:end],
        }

        # Compute next observations for the episode
        episode_data['next_observations'] = np.roll(episode_data['observations'], -1, axis=0)
        terminal_mask = episode_data['terminals'].astype(bool)
        episode_data['next_observations'][terminal_mask] = 0.0

        return episode_data
