from gymnasium.envs.registration import register

from env import NuplanEnv
from config import get_nuplan_config
from dataset import NuplanDataset
from loader import NuplanLoader

# Register nuplan environment
register(
    id='nuplan-v0',
    entry_point='ogbench.nuplan.env:NuplanEnv',
    max_episode_steps=1000,
    kwargs={'config': get_nuplan_config()}
)

register(
    id='visual-nuplan-v0',
    entry_point='ogbench.nuplan.env:NuplanEnv',
    max_episode_steps=1000,
    kwargs={'config': get_nuplan_config(), 'visual': True}
)

__all__ = ['NuplanEnv', 'get_nuplan_config', 'NuplanDataset', 'NuplanLoader']
