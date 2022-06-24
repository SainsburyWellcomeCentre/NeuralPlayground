import numpy as np
import os


def check_crossing_wall(pre_state, new_state, wall, wall_closenes=1e-5):
    """

    Parameters
    ----------
    pre_state : (2,) 2d-ndarray
        2d position of pre-movement
    new_state : (2,) 2d-ndarray
        2d position of post-movement
    wall : (2, 2) ndarray
        [[x1, y1], [x2, y2]] where (x1, y1) is on limit of the wall, (x2, y2) second limit of the wall
    wall_closenes : float
        how close the agent is allowed to be from the wall

    Returns
    -------
    new_state: (2,) 2d-ndarray
        corrected new state. If it is not crossing the wall, then the new_state stays the same, if the state cross the
        wall, new_state will be corrected to a valid place without crossing the wall
    cross_wall: bool
        True if the change in state cross a wall
    """

    # Check if the line of the wall and the line between the states cross
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


def inheritors(klass):
    subclasses = set()
    work = [klass]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return subclasses


def check_directory(dir, create=True):
    check = os.path.isdir(dir)
    if not check and create:
        os.makedirs(dir)
    return check
