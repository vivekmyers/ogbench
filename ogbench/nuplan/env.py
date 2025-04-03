
import gymnasium
import numpy as np
from gymnasium.spaces import Box, Discrete
from PIL import Image


class DrivingEnv(gymnasium.Env):
    """Wrapper around NuPlan driving simulator."""

    metadata = {
        'render_modes': ['rgb_array'],
        'render_fps': 15,
    }

    def __init__(
        self,
        visual=False,
        # world_size=32,
        # grid_size=4,
        # brush_size=4,
        # num_elems=5,
        # mode='task',
        # render_mode=None,
        # width=192,
        # height=192,
    ):
        """Initialize the Powderworld environment.

        Args:
            visual: use image observations
        """
        self._visual = visual

    def reset(self, *, seed=None, options=None):
        ob = ...
        info = ...

        return ob, info

    def step(self, action):
        ob = self._get_ob()
        reward = ...
        terminated = ...
        info = {
            ...: ...
        }

        return ob, reward, terminated, False, info

    def render(self):
        ob = self._get_ob()
        frame = ob[..., :3]
        frame = Image.fromarray(frame)
        frame = frame.resize((self._render_width, self._render_height), Image.NEAREST)
        frame = np.array(frame)

        return frame

    def _get_ob(self):
        world_frame = ...
        action_frame = ...
        return np.concatenate([world_frame, action_frame], axis=2)  # (world_size, world_size, 6)
