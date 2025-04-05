from typing import Any, Dict, Optional

import numpy as np


class NuPlanDataset:
    """Minimal NuPlan dataset handler that just adds goal fields."""

    def __init__(self, data_path):
        """Load data and add required goal fields.

        Args:
            data_path: Path to the .npz file containing the dataset
        """
        # Load raw data
        data = dict(np.load(data_path))
        
        # Add goal fields (using next_observations for both)
        data['value_goals'] = data['next_observations'].copy()
        data['actor_goals'] = data['next_observations'].copy()
        
        self.data = data

    def _process_data(self) -> None:
        """Process the dataset arrays to provide exactly what main.py needs."""
        # Convert everything to float32
        self.data['observations'] = self.data['observations'].astype(np.float32)
        self.data['actions'] = self.data['actions'].astype(np.float32)

        # Add rewards if not present (using zeros)
        if 'rewards' not in self.data:
            self.data['rewards'] = np.zeros(len(self.data['observations']), dtype=np.float32)

        # Compute next observations (roll the observations array)
        self.data['next_observations'] = np.roll(self.data['observations'], -1, axis=0)

        # For CRL, use next observations as goals
        self.data['value_goals'] = self.data['next_observations'].copy()
        self.data['actor_goals'] = self.data['next_observations'].copy()

        # Handle terminal states
        terminal_mask = self.data['terminals'].astype(bool)
        self.data['next_observations'][terminal_mask] = 0.0
        self.data['value_goals'][terminal_mask] = 0.0
        self.data['actor_goals'][terminal_mask] = 0.0

    @classmethod
    def load(cls, path: str, config: Optional[Dict[str, Any]] = None) -> 'NuPlanDataset':
        """Load dataset from file."""
        return cls(path)

    def save(self, path: str) -> None:
        """Save dataset to file.

        Args:
            path: Path to save the dataset
        """
        np.savez(path, **self.data)

    def get_subset(self, indices: np.ndarray) -> 'NuPlanDataset':
        """Get a subset of the dataset.

        Args:
            indices: Indices to select

        Returns:
            New NuPlanDataset instance with selected data
        """
        subset_data = {k: v[indices] for k, v in self.data.items()}
        return NuPlanDataset(subset_data)

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
