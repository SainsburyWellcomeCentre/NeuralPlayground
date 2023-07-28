import numpy as np

from ..experiments.wernle_2018_data import Wernle2018Data
from .hafting_2008 import Hafting2008
from .simple2d import Simple2D


class Wernle2018(Hafting2008):
    """Arena resembling Wernle2018 experimental setting

    Methods (In addition to Hafting 2008)
    ----------
    set_room(self, room_id: str):
        Place the agent in the right room configuration depending the amount of exploration time
    def step(self, action: np.ndarray, normalize_step: bool = False, skip_every: int = 10):
        Move the agents based on the action and room configuration

    Attributes (In addition to Hafting 2008)
    ----------
    merge_time: float
        Time in minutes to remove the middle wall in the experiment, merging the two rooms
    switch_time: float
        Time in minutes to change the agent from one room to the other one
    """

    def __init__(
        self,
        use_behavioral_data: bool = False,
        data_path: str = None,
        recording_index: int = None,
        environment_name: str = "Wernle2018",
        verbose: bool = False,
        experiment_class=Wernle2018Data,
        merge_time: float = 100,
        switch_time: float = 50,
        **env_kwargs,
    ):
        """

        Parameters
        ----------
        use_behavioral_data: bool
            If True, then uses the animal trajectories recorded in Wernle 2018
        data_path: str
            if None, fetch the data from the NeuralPlayground data repository,
            else load data from given path
        recording_index: int
            if None, load data from default recording index of corresponding experiment class
        environment_name: str
            Name of the specific instantiation of the Wernle 2018 class
        verbose: bool
            Set to True to show the information of the class
        experiment_class:
            Experiment class to be initialized
        merge_time: float
            Time in minutes to remove the middle wall in the experiment, merging the two rooms
        switch_time: float
            Time in minutes to change the agent from one room to the other one
        env_kwargs: dict (keys below)
            time_step_size: float
                Time step size in second
        """
        super().__init__(
            use_behavioral_data,
            data_path,
            recording_index,
            environment_name,
            verbose,
            experiment_class,
            **env_kwargs,
        )
        self.time_step_size = env_kwargs["time_step_size"]
        self.merge_time = int((merge_time * 60) / self.time_step_size)
        self.switch_time = int((switch_time * 60) / self.time_step_size)

        self.AB_id = "AB"
        self.A_id = "A"
        self.B_id = "B"

    def set_room(self, room_id: str):
        """Place the agent in the right room configuration depending the amount of exploration time

        Parameters
        ----------
        room_id: str
            Either "A", "B" or "AB". If "A" or "B" it will locate the agent in one of the rooms with a wall in between.
            If "AB", it will place the agent in a random position and remove the wall in between
        """
        if room_id == self.A_id:
            # Take the agent to room A
            self.state = [
                np.random.uniform(low=self.arena_limits[0, 0], high=self.arena_limits[0, 1]),
                np.random.uniform(low=self.arena_limits[1, 0], high=0),
            ]
            # Add wall in between to separete rooms
            self.custom_walls = [np.array([[-100, 0], [100, 0]])]
            # Update walls in the environment
            self.wall_list = self.default_walls + self.custom_walls
        elif room_id == self.B_id:
            # Take the agent to room B
            self.state = [
                np.random.uniform(low=self.arena_limits[0, 0], high=self.arena_limits[0, 1]),
                np.random.uniform(low=0, high=self.arena_limits[1, 1]),
            ]
            # Add wall in between to separete rooms
            self.custom_walls = [np.array([[-100, 0], [100, 0]])]
            # Update walls in the environment
            self.wall_list = self.default_walls + self.custom_walls
        elif room_id == self.AB_id:
            if len(self.custom_walls) != 0:
                # Remove wall in between to merge rooms
                self.custom_walls = []
                # Update walls in the environment
                self.wall_list = self.default_walls + self.custom_walls

    def _create_custom_walls(self):
        """Add wall in between when initializing environment as default"""
        self.custom_walls = [np.array([[-100, 0], [100, 0]])]

    def step(self, action: np.ndarray, normalize_step: bool = False, skip_every: int = 10):
        """Set the right room configuration, then call default step function"""
        if self.use_behavioral_data:
            return super().step(action, normalize_step, skip_every)
        if self.global_steps == 0:
            self.set_room("A")
        elif self.global_steps == self.merge_time:
            self.set_room("AB")
        elif self.global_steps == self.switch_time:
            self.set_room("B")
        return super().step(action, normalize_step, skip_every)


class MergingRoom(Simple2D):
    """Arena resembling Wernle2018 experimental setting BUT WITH GENERIC ROOM DIMENSIONS

    Methods (In addition to Simple2D)
    ----------
    set_room(self, room_id: str):
        Place the agent in the right room configuration depending the amount of exploration time
    def step(self, action: np.ndarray, normalize_step: bool = False, skip_every: int = 10):
        Move the agents based on the action and room configuration

    Attributes (In addition to Simple2D)
    ----------
    merge_time: float
        Time in minutes to remove the middle wall in the experiment, merging the two rooms
    switch_time: float
        Time in minutes to change the agent from one room to the other one
    """

    def __init__(
        self,
        environment_name: str = "Wernle2018",
        merge_time: float = 100,
        switch_time: float = 50,
        **env_kwargs,
    ):
        """

        Parameters
        ----------
        use_behavioral_data: bool
            If True, then uses the animal trajectories recorded in Wernle 2018
        data_path: str
            if None, fetch the data from the NeuralPlayground data repository,
            else load data from given path
        recording_index: int
            if None, load data from default recording index of corresponding experiment class
        environment_name: str
            Name of the specific instantiation of the Wernle 2018 class
        verbose: bool
            Set to True to show the information of the class
        experiment_class:
            Experiment class to be initialized
        merge_time: float
            Time in minutes to remove the middle wall in the experiment, merging the two rooms
        switch_time: float
            Time in minutes to change the agent from one room to the other one
        env_kwargs: dict (keys below)
            time_step_size: float
                Time step size in second
        """

        super().__init__(environment_name, **env_kwargs)

        self.time_step_size = env_kwargs["time_step_size"]
        self.merge_time = int((merge_time * 60) / self.time_step_size)
        self.switch_time = int((switch_time * 60) / self.time_step_size)

        self.AB_id = "AB"
        self.A_id = "A"
        self.B_id = "B"

    def set_room(self, room_id: str):
        """Place the agent in the right room configuration depending the amount of exploration time

        Parameters
        ----------
        room_id: str
            Either "A", "B" or "AB". If "A" or "B" it will locate the agent in one of the rooms with a wall in between.
            If "AB", it will place the agent in a random position and remove the wall in between
        """
        if room_id == self.A_id:
            # Take the agent to room A
            self.state = [
                np.random.uniform(low=self.arena_limits[0, 0], high=self.arena_limits[0, 1]),
                np.random.uniform(low=self.arena_limits[1, 0], high=0),
            ]
            # Add wall in between to separete rooms
            self.custom_walls = [np.array([[self.arena_limits[0, 0], 0], [self.arena_limits[0, 1], 0]])]
            # Update walls in the environment
            self.wall_list = self.default_walls + self.custom_walls
        elif room_id == self.B_id:
            # Take the agent to room B
            self.state = [
                np.random.uniform(low=self.arena_limits[0, 0], high=self.arena_limits[0, 1]),
                np.random.uniform(low=0, high=self.arena_limits[1, 1]),
            ]
            # Add wall in between to separete rooms
            self.custom_walls = [np.array([[self.arena_limits[0, 0], 0], [self.arena_limits[0, 1], 0]])]
            # Update walls in the environment
            self.wall_list = self.default_walls + self.custom_walls
        elif room_id == self.AB_id:
            if len(self.custom_walls) != 0:
                # Remove wall in between to merge rooms
                self.custom_walls = []
                # Update walls in the environment
                self.wall_list = self.default_walls + self.custom_walls

    def _create_custom_walls(self):
        """Add wall in between when initializing environment as default"""
        self.custom_walls = [np.array([[self.arena_limits[0, 0], 0], [self.arena_limits[0, 1], 0]])]

    def step(self, action: np.ndarray, normalize_step: bool = False):
        """Set the right room configuration, then call default step function"""
        if self.global_steps == 0:
            self.set_room("A")
        elif self.global_steps == self.merge_time:
            self.set_room("AB")
        elif self.global_steps == self.switch_time:
            self.set_room("B")
        return super().step(action, normalize_step)
