import os.path
import numpy as np
import scipy.io as sio
import glob
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter

class FullHaftingData(object):

    def __init__(self, data_path, experiment_name="FullHaftingData", verbose=False, session=None):
        self.experiment_name = experiment_name
        self.data_path = data_path
        self._load_data()
        
        if session is None:
            self.rat_id, self.sess = self.best_session["rat_id"], self.best_session["sess"]
        else:
            self.rat_id = session["rat_id"]
            self.sess = session["sess"]
        if verbose:
            self.show_readme()
            self.show_keys()
            self.plot_session()
        self.set_behavioral_data()

    def _load_data(self):
        self.best_session ={"rat_id": "11015", "sess": "13120410", "cell_id": "t5c1"}
        self.arena_limits = np.array([[-200, 200], [-20, 20]])
        data_path_list = glob.glob(self.data_path + "*.mat")
        mice_ids = np.unique([dp.split("/")[-1][:5] for dp in data_path_list])
        self.data_per_animal = {}
        for m_id in mice_ids:
            m_paths_list = glob.glob(self.data_path + m_id + "*.mat")
            sessions = np.unique([dp.split("/")[-1].split("-")[1][:8] for dp in m_paths_list]).astype(str)
            self.data_per_animal[m_id] = {}
            for sess in sessions:
                s_paths_list = glob.glob(self.data_path + m_id + "-" + sess + "*.mat")
                cell_ids = np.unique([dp.split("/")[-1].split(".")[-2][-4:] for dp in s_paths_list]).astype(str)
                self.data_per_animal[m_id][sess] = {}
                for cell_id in cell_ids:
                    if cell_id == "_POS":
                        session_info = "position"
                    elif cell_id in ["_EEG", "_EGF"]:
                        continue
                    else:
                        session_info = cell_id
                        print(cell_id)
                    r_path = glob.glob(self.data_path + m_id + "-" + sess + "*" + cell_id + "*.mat")
                    cleaned_data = clean_data(sio.loadmat(r_path[0]))
                    self.data_per_animal[m_id][sess][session_info] = cleaned_data

    def show_keys(self):
        print("Rat ids", list(self.data_per_animal.keys()))
        for rat_id, val in self.data_per_animal.items():
            print("Sessions for " + rat_id)
            print(list(self.data_per_animal[rat_id].keys()))
            for sess in self.data_per_animal[rat_id].keys():
                print("Cells recorded in session " + sess)
                print(list(self.data_per_animal[rat_id][sess].keys()))

    def show_readme(self):
        readme_path = glob.glob(self.data_path + "readme" + "*.txt")[0]
        with open(readme_path, 'r') as fin:
            print(fin.read())

    def plot_session(self):
        # print(self.data_per_animal[self.best_session["rat"]][self.best_session["sess"]])
        cell_data = self.data_per_animal[self.rat_id][self.sess]
        position_data = cell_data["position"]
        x1, y1 = position_data["posx"][:], position_data["posy"][:]
        # Selecting positional data
        x = np.clip(x1, a_min=-200, a_max=200)
        y = np.clip(y1, a_min=-20, a_max=20)
        
        # Selecting positional data
        time_array = position_data["post"][:]
        tetrode_data = cell_data["t5c1"]
        test_spikes = tetrode_data["ts"][:]
        test_spikes = test_spikes[:, 0]
        time_array = time_array[:, 0]
        x = x[:, 0]
        y = y[:, 0]
        
        
        f, ax = plt.subplots(1, 2, figsize=(15, 8))
        ax = ax.flatten()
        ax[0].plot(x, y)
        ax[0].set_title("position")
        h, binx, biny = get_2D_ratemap(time_array, test_spikes, x, y)
        ax[1].imshow(h)
        plt.show()
        
    def set_behavioral_data(self, rat_id=None, session=None):
        arena_limits = np.array([[-50, 50], [-50, 50]])
        if rat_id is None:
            rat_id = self.rat_id
            session = self.sess

        cell_data = self.data_per_animal[rat_id][session]
        position_data = cell_data["position"]
    
        # Selecting positional data
        x1, y1 = position_data["posx"][:], position_data["posy"][:]
        # Selecting positional data
        x = np.clip(x1, a_min=-200, a_max=200)
        y = np.clip(y1, a_min=-20, a_max=20)
        time_array = position_data["post"][:]

        position = np.stack([x, y], axis=1) # Convert to cm
        head_direction = np.diff(position, axis=0)
        head_direction = head_direction/np.sqrt(np.sum(head_direction**2, axis=1))[..., np.newaxis]
        self.arena_limits = arena_limits
        self.position = position
        self.head_direction = head_direction
        self.time = time_array
               

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

    def __init__(self, data_path, experiment_name="FullSargoliniData", verbose=False, session=None):
        self.data_path = data_path
        self.experiment_name = experiment_name
        self._load_data()
        if session is None:
            self.rat_id, self.sess = self.best_session["rat_id"], self.best_session["sess"]
        else:
            self.rat_id = session["rat_id"]
            self.sess = session["sess"]

        if verbose:
            self.show_readme()
            self.show_keys()
            self.plot_session()
        self.set_behavioral_data()

    def _load_data(self):
        self.best_session = {"rat_id": "11016", "sess": "31010502"}
        # self.best_session = {"rat_id": "10704", "sess": "20060402"}
        self.arena_limits = np.array([[-50.0, 50.0], [-50.0, 50.0]])


        data_path_list = glob.glob(self.data_path + "*.mat")
        mice_ids = np.unique([dp.split("/")[-1][:5] for dp in data_path_list])
        self.data_per_animal = {}
        for m_id in mice_ids:
            m_paths_list = glob.glob(self.data_path + m_id + "*.mat")
            sessions = np.unique([dp.split("/")[-1].split("-")[1][:8] for dp in m_paths_list]).astype(str)
            self.data_per_animal[m_id] = {}
            for sess in sessions:
                s_paths_list = glob.glob(self.data_path + m_id + "-" + sess + "*.mat")
                cell_ids = np.unique([dp.split("/")[-1].split(".")[-2][-4:] for dp in s_paths_list]).astype(str)
                self.data_per_animal[m_id][sess] = {}
                for cell_id in cell_ids:
                    if cell_id == "_POS":
                        session_info = "position"
                    elif cell_id in ["_EEG", "_EGF"]:
                        continue
                    else:
                        session_info = cell_id

                    r_path = glob.glob(self.data_path + m_id + "-" + sess + "*" + cell_id + "*.mat")
                    cleaned_data = clean_data(sio.loadmat(r_path[0]))
                    if cell_id != "_POS":
                        try:
                            self.data_per_animal[m_id][sess][session_info] = cleaned_data["cellTS"]
                        except:
                            pass
                    else:
                        self.data_per_animal[m_id][sess][session_info] = cleaned_data

    def show_keys(self):
        print("Rat ids", list(self.data_per_animal.keys()))
        for rat_id, val in self.data_per_animal.items():
            print("Sessions for " + rat_id)
            print(list(self.data_per_animal[rat_id].keys()))
            for sess in self.data_per_animal[rat_id].keys():
                print("Cells recorded in session " + sess)
                print(list(self.data_per_animal[rat_id][sess].keys()))

    def show_readme(self):
        readme_path = glob.glob(self.data_path + "readme" + "*.txt")[0]
        with open(readme_path, 'r') as fin:
            print(fin.read())

    def plot_session(self):
        # print(self.data_per_animal[self.best_session["rat"]][self.best_session["sess"]])
        cell_data = self.data_per_animal[self.rat_id][self.sess]
        position_data = cell_data["position"]
        x1, y1 = position_data["posx"][:, 0], position_data["posy"][:, 0]
        if len(position_data["posy2"]) != 0:
            x2, y2 = position_data["posy2"][:, 0], position_data["posy2"][:, 0]
        else:
            x2, y2 = x1, y1

        # Selecting positional data
        x = np.mean(np.stack([x1, x2], axis=1), axis=1)
        y = np.mean(np.stack([y1, y2], axis=1), axis=1)
        x = np.clip(x, a_min=self.arena_limits[0, 0], a_max=self.arena_limits[0, 1])
        y = np.clip(y, a_min=self.arena_limits[1, 0], a_max=self.arena_limits[1, 1])

        time_array = position_data["post"][:, 0]
        f, ax = plt.subplots(2, 3, figsize=(15, 8))
        ax = ax.flatten()
        ax[0].plot(x, y)
        ax[0].set_title("position")

        count_i = 0
        for i, (key, single_cell_data) in enumerate(cell_data.items()):
            if key == "position":
                continue
            test_spikes = single_cell_data[:, 0]
            h, binx, biny = get_2D_ratemap(time_array, test_spikes, x, y, filter_result=True)
            ax[count_i + 1].imshow(h)
            ax[count_i + 1].set_title(key)
            count_i += 1
            if count_i >= 5:
                break
        plt.show()

    def set_behavioral_data(self, rat_id=None, session=None):
        arena_limits = np.array([[-50, 50], [-50, 50]])
        if rat_id is None:
            rat_id = self.rat_id
            session = self.sess

        cell_data = self.data_per_animal[rat_id][session]
        position_data = cell_data["position"]
        x1, y1 = position_data["posx"][:, 0], position_data["posy"][:, 0]
        if len(position_data["posy2"]) != 0:
            x2, y2 = position_data["posy2"][:, 0], position_data["posy2"][:, 0]
        else:
            x2, y2 = x1, y1

        # Selecting positional data
        x = np.mean(np.stack([x1, x2], axis=1), axis=1)
        y = np.mean(np.stack([y1, y2], axis=1), axis=1)
        x = np.clip(x, a_min=self.arena_limits[0, 0], a_max=self.arena_limits[0, 1])
        y = np.clip(y, a_min=self.arena_limits[1, 0], a_max=self.arena_limits[1, 1])
        time_array = position_data["post"][:, 0]

        position = np.stack([x, y], axis=1) # Convert to cm
        head_direction = np.diff(position, axis=0)
        head_direction = head_direction/np.sqrt(np.sum(head_direction**2, axis=1))[..., np.newaxis]
        self.arena_limits = arena_limits
        self.position = position
        self.head_direction = head_direction
        self.time = time_array


def clean_data(data, keep_headers=False):
    aux_dict = {}
    for key, val in data.items():
        if isinstance(val, bytes) or isinstance(val, str) or key == "__globals__":
            if keep_headers:
                aux_dict[key] = val
            continue
        else:
            # print(len(val))
            if not np.isnan(val).any():
                aux_dict[key] = val
            else:
                # Interpolate nans
                x_range = np.linspace(0, 1, num=len(val))
                nan_indexes = np.logical_not(np.isnan(val))[:, 0]
                clean_x = x_range[nan_indexes]
                clean_val = np.array(val)[nan_indexes, 0]
                # print(clean_x.shape, clean_val.shape)
                f = interp1d(clean_x, clean_val, kind='cubic', fill_value="extrapolate")
                aux_dict[key] = f(x_range)[..., np.newaxis]
    return aux_dict


def get_2D_ratemap(time_array, spikes, x, y, x_size=50, y_size=50, filter_result=False):
    x_spikes, y_spikes = [], []
    for s in spikes:
        array_pos = np.argmin(np.abs(time_array-s))
        x_spikes.append(x[array_pos])
        y_spikes.append(y[array_pos])
    x_spikes = np.array(x_spikes)
    y_spikes = np.array(y_spikes)
    h, binx, biny = np.histogram2d(x_spikes, y_spikes, bins=(x_size, y_size))
    if filter_result:
        h = gaussian_filter(h, sigma=2)

    return h.T, binx, biny


if __name__ == "__main__":
    # data_path = "sargolini2006/"
    # experiment_name = "Sargolini2006"
    # data = SargoliniData(data_path=data_path, experiment_name=experiment_name)
    # print(data.position.shape)
    # print(data.head_direction.shape)
    # print(np.amin(data.position), np.amax(data.position))

    data_path = "/home/rodrigo/HDisk/8F6BE356-3277-475C-87B1-C7A977632DA7_1/all_data/"
    data = FullSargoliniData(data_path=data_path, verbose=True)
