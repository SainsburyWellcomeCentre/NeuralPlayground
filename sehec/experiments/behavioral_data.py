import os.path
import numpy as np
import scipy.io as sio
import glob
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter
import sys
import sehec
import matplotlib as mpl
from IPython.display import display


class FullHaftingData(object):

    def __init__(self, data_path=None, recording_index=None, experiment_name="FullHaftingData", verbose=False,):
        self.experiment_name = experiment_name
        self._find_data_path(data_path)
        self._load_data()
        self._create_dataframe()
        self.rat_id, self.sess, self.rec_vars = self.get_recorded_session(recording_index)
        if verbose:
            self.show_readme()
            self.show_keys()

    def _find_data_path(self, data_path):
        if data_path is None:
            self.data_path = os.path.join(sehec.__path__[0], "experiments/hafting_2008/")
        else:
            self.data_path = data_path

    def _load_data(self):
        self.best_recording_index = 4
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
                    elif "EG" in cell_id:
                        session_info = cell_id[1:]
                    else:
                        session_info = cell_id

                    r_path = glob.glob(self.data_path + m_id + "-" + sess + "*" + cell_id + "*.mat")
                    cleaned_data = clean_data(sio.loadmat(r_path[0]))
                    self.data_per_animal[m_id][sess][session_info] = cleaned_data

    def _create_dataframe(self):
        self.list = []
        l = 0
        for rat_id, rat_sess in self.data_per_animal.items():
            for sess, recorded_vars in rat_sess.items():
                self.list.append( {"rec_index": l, "rat_id": rat_id, "session": sess,
                                   "recorded_vars": list(recorded_vars.keys())})
                l += 1
        self.recording_list = pd.DataFrame(self.list).set_index("rec_index")

    def show_keys(self):
        print("Dataframe with recordings")
        display(self.recording_list)

    def show_readme(self):
        readme_path = glob.glob(self.data_path + "readme" + "*.txt")[0]
        with open(readme_path, 'r') as fin:
            print(fin.read())

    def get_recorded_session(self, recording_index=None):
        if recording_index is None:
            recording_index = self.best_recording_index
        list_item = self.recording_list.iloc[recording_index]
        rat_id, sess, recorded_vars = list_item["rat_id"], list_item["session"], list_item["recorded_vars"]
        return rat_id, sess, recorded_vars

    def get_recording_data(self, recording_index=None):
        if recording_index is None:
            recording_index = self.best_recording_index
        if type(recording_index) is list or type(recording_index) is tuple:
            data_list = []
            for ind in recording_index:
                rat_id, sess, rec_vars = self.get_recorded_session(ind)
                session_data = self.data_per_animal[rat_id][sess]
                data_list.append(session_data)
            return data_list
        else:
            rat_id, sess, rec_vars = self.get_recorded_session(recording_index)
            session_data = self.data_per_animal[rat_id][sess]
            return session_data, rec_vars

    def _find_tetrode(self, rev_vars):
        tetrode_id = next(
            var_name for var_name in rev_vars if (var_name != 'position') and (("t" in var_name) or ("T" in var_name)))
        return tetrode_id

    def get_tetrode_data(self, session_data, tetrode_id):
        position_data = session_data["position"]
        x1, y1 = position_data["posx"][:, 0], position_data["posy"][:, 0]
        x2, y2 = x1, y1
        # Selecting positional data
        x = np.clip(x2, a_min=self.arena_limits[0, 0], a_max=self.arena_limits[0, 1])
        y = np.clip(y2, a_min=self.arena_limits[1, 0], a_max=self.arena_limits[1, 1])
        time_array = position_data["post"][:]
        tetrode_data = session_data[tetrode_id]
        test_spikes = tetrode_data["ts"][:, ]
        test_spikes = test_spikes[:, 0]
        time_array = time_array[:, 0]
        return time_array, test_spikes, x, y

    def plot_recording_tetr(self, recording_index=None, save_path=None, ax=None, tetrode_id=None):
        session_data, rev_vars = self.get_recording_data(recording_index)
        if tetrode_id is None:
            tetrode_id = self._find_tetrode(rev_vars)

        arena_width = self.arena_limits[0, 1] - self.arena_limits[0, 0]
        arena_depth = self.arena_limits[1, 1] - self.arena_limits[1, 0]

        time_array, test_spikes, x, y = self.get_tetrode_data(session_data, tetrode_id)

        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(10, 8))

        scale_ratio = 2  # To discretize space
        h, binx, biny = get_2D_ratemap(time_array, test_spikes, x, y, x_size=int(arena_width/scale_ratio),
                                       y_size=int(arena_depth/scale_ratio), filter_result=True)
        sc = ax.imshow(h, cmap='jet')
        cbar = plt.colorbar(sc, ax=ax, ticks=[np.min(h), np.max(h)], orientation="horizontal")
        cbar.ax.set_xlabel('Firing rate', fontsize=12)
        cbar.ax.set_xticklabels([np.round(np.min(h)), np.round(np.max(h))], fontsize=12)
        ax.set_title(tetrode_id)
        ax.set_ylabel('width', fontsize=16)
        ax.set_xlabel('depth', fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])

        if not save_path is None:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close("all")
        else:
            return ax

    def plot_trajectory(self, recording_index=None, save_path=None, ax=None, plot_every=20):
        session_data, rev_vars = self.get_recording_data(recording_index)
        tetrode_id = self._find_tetrode(rev_vars)


        time_array, test_spikes, x, y = self.get_tetrode_data(session_data, tetrode_id)
        print("debug")

        arena_width = self.arena_limits[0, 1] - self.arena_limits[0, 0]
        arena_depth = self.arena_limits[1, 1] - self.arena_limits[1, 0]
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(10, 8))

        ax.plot([self.arena_limits[0, 0], self.arena_limits[0, 0]],
                [self.arena_limits[1, 0], self.arena_limits[1, 1]], "C3", lw=3)
        ax.plot([self.arena_limits[0, 1], self.arena_limits[0, 1]],
                [self.arena_limits[1, 0], self.arena_limits[1, 1]], "C3", lw=3)
        ax.plot([self.arena_limits[0, 0], self.arena_limits[0, 1]],
                [self.arena_limits[1, 1], self.arena_limits[1, 1]], "C3", lw=3)
        ax.plot([self.arena_limits[0, 0], self.arena_limits[0, 1]],
                [self.arena_limits[1, 0], self.arena_limits[1, 0]], "C3", lw=3)

        cmap = mpl.cm.get_cmap("plasma")
        norm = plt.Normalize(0, np.size(x))

        aux_x = []
        aux_y = []
        for i in range(len(x)):
            if i % plot_every == 0:
                if i + plot_every >= len(x):
                    break
                x_ = [x[i], x[i + plot_every]]
                y_ = [y[i], y[i + plot_every]]
                aux_x.append(x[i])
                aux_y.append(y[i])
                sc = ax.plot(x_, y_, "-", color=cmap(norm(i)), alpha=0.6, linewidth=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('width')
        ax.set_xlabel('depth')
        ax.set_title("position")

        cmap = mpl.cm.get_cmap("plasma")
        norm = plt.Normalize(0, np.size(x))
        sc = ax.scatter(aux_x, aux_y, c=np.arange(len(aux_x)), vmin=0, vmax=len(x), cmap="plasma", alpha=0.6, s=0.1)

        cbar = plt.colorbar(sc, ax=ax, ticks=[0, len(x)])
        cbar.ax.set_ylabel('N steps', rotation=270, fontsize=12)
        cbar.ax.set_yticklabels([0, len(x)], fontsize=12)
        ax.set_xlim([np.amin([x.min(), y.min()])-1.0, np.amax([x.max(), y.max()])+1.0])
        ax.set_ylim([np.amin([x.min(), y.min()])-1.0, np.amax([x.max(), y.max()])+1.0])


class SargoliniData(object):

    def __init__(self, experiment_name='Sargolini_2006_Data', data_path=None):
        if data_path is None:
            self.data_path = os.path.join(sehec.__path__[0], "experiments/sargolini_2006")
        else:
            self.data_path = data_path
        self.arena_limits, self.position, self.head_direction = self.get_sargolini_data()

    def get_sargolini_data(self, tolerance=1e-10):
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
        head_direction = head_direction/np.sqrt(np.sum(head_direction**2, axis=1) + tolerance)[..., np.newaxis]
        return arena_limits, position, head_direction


class FullSargoliniData(FullHaftingData):

    def __init__(self, data_path=None, recording_index=None, experiment_name="FullSargoliniData", verbose=False):
        super().__init__(data_path=data_path, recording_index=recording_index,
                         experiment_name=experiment_name, verbose=verbose)

    def _find_data_path(self, data_path):
        if data_path is None:
            self.data_path = os.path.join(sehec.__path__[0], "experiments/sargolini_2006/raw_data_sample/")
        else:
            self.data_path = data_path

    def _load_data(self):
        self.best_recording_index = 0
        # self.best_session = {"rat_id": "10704", "session": "20060402"}
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
                        session_info = cell_id[1:]
                    else:
                        session_info = cell_id
                    r_path = glob.glob(self.data_path + m_id + "-" + sess + "*" + cell_id + "*.mat")
                    cleaned_data = clean_data(sio.loadmat(r_path[0]))
                    if cell_id != "_POS" and not cell_id in ["_EEG", "_EGF"]:
                        try:
                            self.data_per_animal[m_id][sess][session_info] = cleaned_data["cellTS"]
                        except:
                            pass
                    else:
                        self.data_per_animal[m_id][sess][session_info] = cleaned_data

    def get_tetrode_data(self, session_data, tetrode_id):
        position_data = session_data["position"]
        x1, y1 = position_data["posx"][:, 0], position_data["posy"][:, 0]
        x2, y2 = x1, y1
        # Selecting positional data
        x = np.clip(x2, a_min=self.arena_limits[0, 0], a_max=self.arena_limits[0, 1])
        y = np.clip(y2, a_min=self.arena_limits[1, 0], a_max=self.arena_limits[1, 1])
        time_array = position_data["post"][:]
        tetrode_data = session_data[tetrode_id]
        test_spikes = tetrode_data[:, 0]
        time_array = time_array[:, 0]
        return time_array, test_spikes, x, y


def clean_data(data, keep_headers=False):
    aux_dict = {}
    for key, val in data.items():
        if isinstance(val, bytes) or isinstance(val, str) or key == "__globals__":
            if keep_headers:
                aux_dict[key] = val
            continue
        else:

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
    # print("initializing hafting")
    # data = FullHaftingData(verbose=True)
    # print("plotting_tragectory")
    # data.plot_trajectory(2)
    # print("plotting_recording")
    # data.plot_recording_tetr(2)
    # plt.show()

    print("initializing sargolini")
    data = FullSargoliniData(verbose=True)
    print("plotting_tragectory")
    data.plot_trajectory(2)
    print("plotting_recording")
    data.plot_recording_tetr(2)
    plt.show()







