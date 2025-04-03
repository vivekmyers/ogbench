from gymnasium.envs.registration import register

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
