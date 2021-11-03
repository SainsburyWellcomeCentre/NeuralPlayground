import os.path
import numpy as np
import scipy.io as sio
import glob
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd

class SargoliniData(object):

    def __init__(self, data_path, experiment_name):
        self.data_path = data_path
        self.experiment_name = experiment_name
        if self.experiment_name == "Sargolini2006":
            self.arena_limits, self.position, self.head_direction = self.get_sargolini_data()

    def get_sargolini_data(self):
        arena_limits = np.array([[-50, 50], [-50, 50]])
        filenames_x = os.path.join(self.data_path, "sargolini_x_pos_")
        filenames_y = os.path.join(self.data_path, "sargolini_y_pos_")

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


class FullSargoliniData(object):

    def __init__(self, data_path, experiment_name = "FullSargoliniData"):
        self.data_path = data_path
        self.experiment_name = experiment_name
        self._load_data()

    def _load_data(self):
        data_path_list = glob.glob(data_path + "*.mat")
        mice_ids = np.unique([dp.split("/")[-1][:5] for dp in data_path_list])
        self.data_per_animal = {}
        for m_id in mice_ids:
            m_paths_list = glob.glob(data_path + m_id + "*.mat")
            sessions = np.unique([dp.split("/")[-1].split("-")[1][:8] for dp in m_paths_list]).astype(str)
            self.data_per_animal[m_id] = {}
            for sess in sessions:
                s_paths_list = glob.glob(data_path + m_id + "-" + sess + "*.mat")
                cell_ids = np.unique([dp.split("/")[-1].split(".")[-2][-4:] for dp in s_paths_list]).astype(str)
                self.data_per_animal[m_id][sess] = {}
                for cell_id in cell_ids:
                    r_path = glob.glob(data_path + m_id + "-" + sess + "*" + cell_id + "*.mat")
                    cleaned_data = clean_data(sio.loadmat(r_path[0]))
                    self.data_per_animal[m_id][sess][cell_id] = cleaned_data


def clean_data(data, keep_headers=False):
    aux_dict = {}
    for key, val in data.items():
        if isinstance(val, bytes) or isinstance(val, str) or key=="__globals__":
            if keep_headers:
                aux_dict[key] = val
            continue
        else:
            # print(len(val))
            if np.isnan(val).any():
                aux_dict[key] = val
            else:
                # Interpolate nans
                x_range = np.linspace(0, 1, num=len(val))
                nan_indexes = np.logical_not(np.isnan(val))[:, 0]
                clean_x = x_range[nan_indexes]
                clean_val = np.array(val)[nan_indexes, 0]
                # print(clean_x.shape, clean_val.shape)
                f = interp1d(clean_x, clean_val, kind='cubic')
                aux_dict[key] = f(x_range)
    return aux_dict


if __name__ == "__main__":
    # data_path = "sargolini2006/"
    # experiment_name = "Sargolini2006"
    # data = SargoliniData(data_path=data_path, experiment_name=experiment_name)
    # print(data.position.shape)
    # print(data.head_direction.shape)
    # print(np.amin(data.position), np.amax(data.position))

    data_path = "/home/rodrigo/HDisk/8F6BE356-3277-475C-87B1-C7A977632DA7_1/"
    data = FullSargoliniData(data_path=data_path)