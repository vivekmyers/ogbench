from gymnasium.envs.registration import register
from ogbench.nuplan.env import NuPlanEnv
from ogbench.nuplan.process_data import process_nuplan_data

# Register the environment
register(
    id='nuplan-v0',
    entry_point='ogbench.nuplan.env:NuPlanEnv',
    max_episode_steps=100,
)

__all__ = ['NuPlanEnv', 'process_nuplan_data']
