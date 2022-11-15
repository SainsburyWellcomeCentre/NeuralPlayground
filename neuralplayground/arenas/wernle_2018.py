import numpy as np
from .simple2d import Simple2D
from ..experiments.wernle_2018_data import Wernle2018Data


class Wernle2018(Simple2D):
    def __init__(self, environment_name="MergingRoom", merge_time=100, switch_time=50, verbose=False, data_path=None, **env_kwargs):
        self.environment_name = environment_name
        env_kwargs["arena_x_limits"] = np.array([-100, 100])
        env_kwargs["arena_y_limits"] = np.array([-100, 100])
        self.time_step_size = env_kwargs["time_step_size"]
        self.merge_time = (merge_time*60) / self.time_step_size
        self.switch_time = (switch_time*60) / self.time_step_size
        self.run_full_experiment = True
        self.data_path = data_path
        self.data = Wernle2018Data(data_path=self.data_path, verbose=verbose)
        super().__init__(environment_name, **env_kwargs)
        self.AB_id = "AB"
        self.A_id = "A"
        self.B_id = "B"

    def set_room(self, room_id):
        if room_id == self.A_id:
            self.state = [np.random.uniform(low=self.arena_limits[0, 0], high=self.arena_limits[0, 1]),
                          np.random.uniform(low=self.arena_limits[1, 0], high=0)]
            self.custom_walls = [np.array([[-100, 0], [100, 0]])]
            self.wall_list = self.default_walls + self.custom_walls
        elif room_id == self.B_id:
            self.state = [np.random.uniform(low=self.arena_limits[0, 0], high=self.arena_limits[0, 1]),
                          np.random.uniform(low=0, high=self.arena_limits[1, 1])]
        elif room_id == self.AB_id:
            if len(self.custom_walls) != 0:
                self.custom_walls = []
                self.wall_list = self.default_walls + self.custom_walls

    def _create_custom_walls(self):
        self.custom_walls = [np.array([[-100, 0], [100, 0]])]

    def step(self, action):
        if self.run_full_experiment:
            if self.global_steps == 0:
                self.set_room("A")
            elif self.global_steps == self.merge_time:
                self.set_room("AB")
            elif self.global_steps == self.switch_time:
                self.set_room("B")

        return super().step(action)