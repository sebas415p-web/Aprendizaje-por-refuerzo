import gymnasium as gym
from gymnasium import spaces
import numpy as np
from environment import (TwoRoomsEnv, N_ACTIONS, ACTION_NAMES,
                         EMPTY, WALL, KEY, BALL, DOOR_CLOSED, DOOR_OPEN, GOAL)


class TwoRoomsGymEnv(gym.Env):
    """
    Wrapper Gymnasium para TwoRoomsEnv.
    Adapta el ambiente custom a la API estándar de Gymnasium sin modificar los originales.
    """
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self._env = TwoRoomsEnv()

        self.observation_space = spaces.Tuple((
            spaces.Discrete(9),
            spaces.Discrete(5),
            spaces.Discrete(2),
            spaces.Discrete(2),
            spaces.Discrete(2),
        ))
        self.action_space = spaces.Discrete(N_ACTIONS)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self._env.reset()
        info = {}
        if self.render_mode == "ansi":
            return state, {**info, "render": self._env.render()}
        return state, info

    def step(self, action):
        state, reward, done, info = self._env.step(action)
        terminated = done and info.get("event") != "TIMEOUT"
        truncated = done and info.get("event") == "TIMEOUT"
        if self.render_mode == "ansi":
            info["render"] = self._env.render()
        return state, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            return self._env.render()
        return None

    def close(self):
        pass
