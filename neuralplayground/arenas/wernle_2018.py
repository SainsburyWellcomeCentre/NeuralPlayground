import numpy as np
from .hafting_2008 import Hafting2008
from ..experiments.wernle_2018_data import Wernle2018Data


class Wernle2018(Hafting2008):
    def __init__(self, use_behavioral_data: bool = False, data_path: str = None, recording_index: int = None,
                 environment_name: str = "Wernle2018", verbose: bool = False, experiment_class=Wernle2018Data,
                 merge_time: float = 100, switch_time: float = 50, **env_kwargs):
        super().__init__(use_behavioral_data, data_path, recording_index, environment_name, verbose, experiment_class,
                         **env_kwargs)
        self.time_step_size = env_kwargs["time_step_size"]
        self.merge_time = (merge_time*60) / self.time_step_size
        self.switch_time = (switch_time*60) / self.time_step_size

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

    def step(self, action: np.ndarray, normalize_step: bool = False, skip_every: int = 10):
        if self.global_steps == 0:
            self.set_room("A")
        elif self.global_steps == self.merge_time:
            self.set_room("AB")
        elif self.global_steps == self.switch_time:
            self.set_room("B")
        return super().step(action, normalize_step, skip_every)