"""
From https://doi.org/10.1016/j.cub.2015.02.037
"""

import numpy as np
from .simple2d import Simple2D
from ...utils import check_crossing_wall


class ConnectedRooms(Simple2D):
    """
    Simulation from https://doi.org/10.1016/j.cub.2015.02.037
    Grid Cells Form a Global Representation of Connected Environments
    """
    def __init__(self, environment_name="ConnectedRooms", corridor_ysize=40.0, singleroom_ysize=90.0,
                 singleroom_xsize=90, door_size=10.0, **env_kwargs):
        """
        Parameters
        ----------
        environment_name : string
            name of the environment
        corridor_ysize : float
            corridor size from the paper, default 40.0 cm
        singleroom_ysize : float
            y-size of one of the rooms, default 90.0 cm
        singleroom_xsize : float
            x-size of one of the rooms, default 90.0 cm
        door_size : float
            door size from room to corridor, default 10 cm
        env_kwargs
        """

        self.corridor_ysize = corridor_ysize
        self.singleroom_ysize = singleroom_ysize
        self.singleroom_xsize = singleroom_xsize
        self.door_size = door_size

        env_kwargs["room_width"] = 2*singleroom_xsize
        env_kwargs["room_depth"] = corridor_ysize + singleroom_ysize

        super().__init__(environment_name, **env_kwargs)

        self.arena_limits = np.array([[-self.singleroom_xsize, self.singleroom_xsize],
                                      [-self.singleroom_ysize, self.corridor_ysize]])

        self.define_walls()

    def create_default_walls(self):
        self.wall_list.append(np.array([[self.arena_limits[0, 0], self.arena_limits[1, 0]],
                                        [self.arena_limits[0, 0], self.arena_limits[1, 1]]]))
        self.wall_list.append(np.array([[self.arena_limits[0, 1], self.arena_limits[1, 0]],
                                        [self.arena_limits[0, 1], self.arena_limits[1, 1]]]))
        self.wall_list.append(np.array([[self.arena_limits[0, 0], self.arena_limits[1, 0]],
                                        [self.arena_limits[0, 1], self.arena_limits[1, 0]]]))
        self.wall_list.append(np.array([[self.arena_limits[0, 0], self.arena_limits[1, 1]],
                                        [self.arena_limits[0, 1], self.arena_limits[1, 1]]]))

    def define_walls(self):
        self.wall_list = []
        # Walls from limit
        self.create_default_walls()

        self.wall_list.append(np.array([[0, 0], [0, -self.singleroom_ysize]]))
        self.wall_list.append(np.array([[-self.singleroom_xsize, 0], [-(self.singleroom_xsize/2+self.door_size/2), 0]]))
        self.wall_list.append(np.array([[-(self.singleroom_xsize/2-self.door_size/2), 0], [0, 0]]))
        self.wall_list.append(np.array([[0, 0], [self.singleroom_xsize/2-self.door_size/2, 0]]))
        self.wall_list.append(np.array([[self.singleroom_xsize/2+self.door_size/2, 0], [self.singleroom_xsize, 0]]))

    def step(self, action):
        self.global_steps += 1
        action = action/np.linalg.norm(action)
        new_state = self.state + self.agent_step_size*action
        new_state, valid_action = self.validate_action(self.state, action, new_state)
        reward = 0  # If you get reward, it should be coded here
        transition = {"action": action, "state": self.state, "next_state": new_state,
                      "reward": reward, "step": self.global_steps}
        self.history.append(transition)
        self.state = new_state
        observation = self.make_observation()
        return observation, new_state, reward

    def validate_action(self, pre_state, action, new_state):
        """

        Parameters
        ----------
        pre_state : (2,) 2d-ndarray
            2d position of pre-movement
        new_state : (2,) 2d-ndarray
            2d position of post-movement

        Returns
        -------
        new_state: (2,) 2d-ndarray
            corrected new state. If it is not crossing the wall, then the new_state stays the same, if the state cross the
            wall, new_state will be corrected to a valid place without crossing the wall
        valid_action: bool
            True if the change in state cross a wall
        """
        valid_action = True
        for wall in self.wall_list:
            new_state, new_valid_action = check_crossing_wall(pre_state=pre_state, new_state=new_state, wall=wall)
            valid_action = new_valid_action and valid_action
        return new_state, valid_action

    def plot_trajectory(self, history_data=None, ax=None):
        if len(self.wall_list) != 0:
            ax = super().plot_trajectory(history_data=history_data, center_room=False)
            for wall in self.wall_list:
                ax.plot(wall[:, 0], wall[:, 1], "b", lw=3)
            return ax
        else:
            return super().plot_trajectory(history_data=self.history, center_room=False)