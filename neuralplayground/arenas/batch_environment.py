import matplotlib as mpl
import matplotlib.pyplot as plt
from neuralplayground.arenas.arena_core import Environment
from neuralplayground.arenas.simple2d import Simple2D
import numpy as np
from neuralplayground.utils import check_crossing_wall


class BatchEnvironment(Environment):
    def __init__(self, environment_name: str = "BatchEnv", env_class: object = Simple2D, batch_size: int = 1,
                 **env_kwargs):
        super().__init__(environment_name, **env_kwargs)
        self.batch_size = batch_size
        batch_x_limits = env_kwargs['arena_x_limits']
        batch_y_limits = env_kwargs['arena_y_limits']
        self.environments = []
        for i in range(self.batch_size):
            env_kwargs['arena_x_limits'] = batch_x_limits[i]
            env_kwargs['arena_y_limits'] = batch_y_limits[i]
            self.environments.append(env_class(**env_kwargs))

        self.room_widths = [np.diff(self.environments[i].arena_x_limits)[0] for i in range(self.batch_size)]
        self.room_depths = [np.diff(self.environments[i].arena_y_limits)[0] for i in range(self.batch_size)]
        self.state_densities = [self.environments[i].state_density for i in range(self.batch_size)]

    def reset(self, random_state: bool = False, custom_state: np.ndarray = None):
        self.global_steps = 0
        self.global_time = 0
        self.history = []

        all_observations = []
        all_states = []
        for i, env in enumerate(self.environments):
            env_obs, env_state = env.reset()
            all_states.append(env_state)
            all_observations.append(env_obs)

        return all_observations, all_states

    def step(self, actions: np.ndarray, normalize_step: bool = False):
        all_observations = []
        all_states = []
        all_rewards = []
        all_allowed = True
        for batch, env in enumerate(self.environments):
            action = actions[batch]
            env_obs, env_state = env.step(action, normalize_step)
            if env.state[0] == env.old_state[0]:
                all_allowed = False
            all_observations.append(env_obs)
            all_states.append(env_state)

        if not all_allowed:
            for env in self.environments:
                env.state = env.old_state

        return all_observations, all_states
