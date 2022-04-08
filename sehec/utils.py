import numpy as np


def check_crossing_wall(pre_state, new_state, wall, wall_closenes=1e-5):

    A = np.stack([np.diff(wall, axis=0)[0, :], -new_state+pre_state], axis=1)
    b = pre_state - wall[0, :]
    intersection = np.linalg.inv(A) @ b
    smaller_than_one = intersection <= 1
    larger_than_zero = intersection >= 0

    # If condition is true, then the points cross the wall
    cross_wall = np.alltrue(np.logical_and(smaller_than_one, larger_than_zero))
    if cross_wall:
        new_state = (intersection[-1]-wall_closenes)*(new_state-pre_state) + pre_state

    return new_state, cross_wall


class RandomAgent(object):
    def act(self, observation):
        return np.random.normal(scale=0.1, size=(2,))
