from gymnasium.envs.registration import register
from .env import NuPlanEnv

# Register nuplan environment
register(
    id='nuplan-v0',
    entry_point='ogbench.nuplan.env:NuPlanEnv',
    max_episode_steps=1000,
)

__all__ = ['NuPlanEnv']
