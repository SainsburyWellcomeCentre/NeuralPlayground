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
        self.environments = []
        for i in range(self.batch_size):
            self.environments.append(env_class(**env_kwargs))

    def reset(self, random_state: bool = False, custom_state: np.ndarray = None):
        self.global_steps = 0
        self.global_time = 0
        self.history = []

        all_observations = []
        all_states = []
        for i,env in enumerate(self.environments):
            if random_state:
                start_pos = [np.random.uniform(low=env.arena_limits[0, 0], high=env.arena_limits[0, 1]),
                              np.random.uniform(low=env.arena_limits[1, 0], high=env.arena_limits[1, 1])]
            else:
                start_pos = [0, 0]
            start_pos = np.asarray(start_pos)

            if custom_state is not None:
                start_pos = np.array(custom_state)
            # Fully observable environment, make_observation returns the state
            observation = env.make_object_observation(pos=start_pos)
            env.state = observation
            all_states.append(env.state)
            all_observations.append(observation)

        return all_observations, all_states

    def step(self, actions: np.ndarray, normalize_step: bool = False):
        all_observations = []
        all_states = []
        all_rewards = []

        for batch, env in enumerate(self.environments):
            action = actions[batch]
            env_obs, env_state = env.step(action)

            all_observations.append(env_obs)
            all_states.append(env_state)

        return all_observations, all_states
