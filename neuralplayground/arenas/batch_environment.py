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
        self.room_width = np.diff(self.environments[0].arena_x_limits)[0]
        self.room_depth = np.diff(self.environments[0].arena_y_limits)[0]
        self.state_density = self.environments[0].state_density

        # Variables for discretised state space
        self.resolution_w = int(self.state_density * self.room_width)
        self.resolution_d = int(self.state_density * self.room_depth)
        self.x_array = np.linspace(-self.room_width / 2, self.room_width / 2, num=self.resolution_w)
        self.y_array = np.linspace(self.room_depth / 2, -self.room_depth / 2, num=self.resolution_d)
        self.mesh = np.array(np.meshgrid(self.x_array, self.y_array))
        self.xy_combination = np.array(np.meshgrid(self.x_array, self.y_array)).T
        self.ws = int(self.room_width * self.state_density)
        self.hs = int(self.room_depth * self.state_density)
        self.n_states = self.resolution_w * self.resolution_d

    def reset(self, random_state: bool = False, custom_state: np.ndarray = None):
        self.global_steps = 0
        self.global_time = 0
        self.history = []

        all_observations = []
        all_states = []
        for i, env in enumerate(self.environments):
            env_obs, env_state = env.reset()
            # if random_state:
            #     env.state[-1] = [np.random.uniform(low=env.arena_x_limits[0], high=env.arena_x_limits[1]),
            #                   np.random.uniform(low=env.arena_x_limits[0], high=env.arena_y_limits[1])]
            # else:
            #     env.state[-1] = [0, 0]
            # # start_pos = np.asarray(start_pos)
            #
            # if custom_state is not None:
            #     env.state[-1] = np.array(custom_state)
            # # Fully observable environment, make_observation returns the state
            # observation = env.make_object_observation()
            # env.state = observation
            # env.objects = env.generate_objects()
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
