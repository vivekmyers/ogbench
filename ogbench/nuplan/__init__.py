from gymnasium.envs.registration import register

<<<<<<< HEAD
from ogbench.nuplan.env import NuplanEnv
from ogbench.nuplan.config import get_nuplan_config
from ogbench.nuplan.dataset import NuplanDataset

# Register nuplan environment
register(
    id='Nuplan-v0',
    entry_point='ogbench.nuplan.env:NuplanEnv',
    max_episode_steps=1000,
    kwargs={'config': get_nuplan_config()}
)

__all__ = ['NuplanEnv', 'get_nuplan_config', 'NuplanDataset'] 
=======
visual_dict = dict(
    ob_type='pixels',
    render_mode='rgb_array',
    width=64,
    height=64,
    camera_name='back',
)

register(
    id='driving-medium-v0'
    entry_point='ogbench.nuplan.env:DrivingEnv',
    max_episode_steps=1000,
    kwargs=dict(
        visual=False,
    ),
)

register(
    id='visual-driving-medium-v0'
    entry_point='ogbench.nuplan.env:DrivingEnv',
    max_episode_steps=1000,
    kwargs=dict(
        visual=True,
    ),
)
>>>>>>> c9706c7b6ed91c9b36ca6706b84625eeb079a95f
