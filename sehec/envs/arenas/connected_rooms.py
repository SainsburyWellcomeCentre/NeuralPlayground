"""
From https://doi.org/10.1016/j.cub.2015.02.037
"""

import numpy as np
from simple2d import Simple2D

def check_crossing_wall(pre_state, new_state, wall):
    pass

class ConnectedRooms(Simple2D):

    def __init__(self, environment_name="ConnectedRooms", corridor_ysize=40.0, singleroom_ysize=90.0,
                 singleroom_xsize=90, door_size=10.0, **env_kwargs):

        self.corridor_ysize = corridor_ysize
        self.singleroom_ysize = singleroom_ysize
        self.singleroom_xsize = singleroom_xsize
        self.door_size = door_size

        env_kwargs["room_width"] = 2*singleroom_xsize
        env_kwargs["room_depth"] = corridor_ysize + singleroom_ysize

        super().__init__(environment_name, **env_kwargs)

        self.arena_limits = np.array([[-self.singleroom_xsize, self.singleroom_xsize],
                                      [-self.singleroom_ysize, self.corridor_ysize]])

    def define_walls(self):
        self.wall_list = []
        self.wall_list.append(np.array([[0, 0], [0, -self.singleroom_ysize]]))
        self.wall_list.append(np.array([[-self.singleroom_xsize, 0], [-(self.singleroom_xsize/2+self.door_size/2), 0]]))
        self.wall_list.append(np.array([[-(self.singleroom_xsize/2-self.door_size/2), 0], [0, 0]]))
        self.wall_list.append(np.array([[0, 0], [self.singleroom_xsize/2-self.door_size/2, 0]]))
        self.wall_list.append(np.array([[self.singleroom_xsize/2+self.door_size/2, 0], [self.singleroom_xsize, 0]]))

    def check_crossing_wall(self, pre_state, new_state, wall):
        pass

    def validate_action(self, pre_state, action, new_state):
        """ Origin at the corner of the two rooms with the corridor"""
        pass


if __name__ == "__main__":
    wall_list = []
    wall_list.append(np.array([[0, 0], [0, -10]]))
    wall_list.append(np.array([[0, 0], [5, 5]]))
    wall_list.append(np.array([[0, 0], [10, 0]]))

    points_set_1 = [np.arrapy([[]])]
