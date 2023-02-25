import numpy as np
import matplotlib.pyplot as plt
# from neuralplayground.experiments import SargoliniDataTrajectory
# from neuralplayground.experiments import Hafting2008Data
from neuralplayground.experiments import Wernle2018Data

from neuralplayground.utils import check_crossing_wall


def plot_movement(pre_state, new_state, ax, valid):
    ax.plot([pre_state[0], new_state[0]], [pre_state[1], new_state[1]], label=str(valid))


def test_walls():
    wall = np.array([[0, 0],
                     [1, 1]])

    pre_state = np.array([0, 0.5])
    new_state1 = np.array([0.5, 0])
    new_state2 = np.array([0.2, 0.4])

    _, valid1 = check_crossing_wall(pre_state, new_state1, wall)
    _, valid2 = check_crossing_wall(pre_state, new_state2, wall)

    f, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(wall[:, 0], wall[:, 1], "k")
    plot_movement(pre_state, new_state1, ax, valid1)
    plot_movement(pre_state, new_state2, ax, valid2)
    ax.legend()
    plt.show()

    print(valid1, valid2)


if __name__ == "__main__":
    # data = SargoliniDataTrajectory()
    # data = Hafting2008Data()
    #data = Wernle2018Data()
    test_walls()


    print("debug")

