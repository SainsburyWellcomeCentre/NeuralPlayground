import numpy as np
from sehec.models.TEM.parameters import *

pars = default_params()


def policy_act(obs):
    action = np.random.normal(scale=0.1, size=2)
    arrow = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    diff = action - arrow
    dist = np.sum(diff ** 2, axis=1)
    index = np.argmin(dist)
    action = arrow[index]
    direc = direction(action)

    return action, direc


def direction(action):
    # Turns action [x,y] into direction [R,L,U,D]
    x, y = action
    direc = np.zeros(shape=4)
    if x > 0 and y == 0:
        d = 0
        name = 'right'
        direc[d] = 1
    elif x < 0 and y == 0:
        d = 1
        name = 'left'
        direc[d] = 1
    elif x == 0 and y > 0:
        d = 2
        name = 'up'
        direc[d] = 1
    elif x == 0 and y < 0:
        d = 3
        name = 'down'
        direc[d] = 1
    else:
        ValueError('impossible action')

    return direc
