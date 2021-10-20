import os.path
import numpy as np


class BehavioralData(object):

    def __init__(self, data_path, experiment_name):
        self.data_path = data_path
        self.experiment_name = experiment_name
        if self.experiment_name == "Sargolini2006":
            self.arena_limits, self.position, self.head_direction = get_sargolini_data(self.data_path)


def get_sargolini_data(data_path):
    arena_limits = np.array([[-50, 50], [-50, 50]])
    filenames_x = os.path.join(data_path, "sargolini_x_pos_")
    filenames_y = os.path.join(data_path, "sargolini_y_pos_")

    x_position = np.array([])
    y_position = np.array([])
    for i in range(61):
        aux_x = np.load(filenames_x + str(i) + ".npy")
        aux_y = np.load(filenames_y + str(i) + ".npy")
        x_position = np.concatenate([x_position, aux_x])
        y_position = np.concatenate([y_position, aux_y])

    position = np.stack([x_position, y_position], axis=1)*100 # Convert to cm
    head_direction = np.diff(position, axis=0)
    head_direction = head_direction/np.sqrt(np.sum(head_direction**2, axis=1))[..., np.newaxis]
    return arena_limits, position, head_direction


if __name__ == "__main__":
    data_path = "sargolini2006/"
    experiment_name = "Sargolini2006"
    data = BehavioralData(data_path=data_path, experiment_name=experiment_name)
    print(data.position.shape)
    print(data.head_direction.shape)
    print(np.amin(data.position), np.amax(data.position))