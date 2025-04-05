"""NuPlan environment registration."""

import gymnasium
from gymnasium.envs.registration import register

# Import the environment class directly
from .env import NuPlanEnv
from .dataset import NuPlanDataset

# Register the NuPlan environment
try:
    register(
        id='nuplan-v0',
        entry_point='ogbench.nuplan.env:NuPlanEnv',
        max_episode_steps=1000,
    )
except gymnasium.error.Error:
    # Environment already registered
    pass

# Export the classes
__all__ = ['NuPlanEnv', 'NuPlanDataset']
