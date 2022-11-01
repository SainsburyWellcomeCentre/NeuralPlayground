import os.path

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import sehec
import warnings
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.interpolate import interp1d
from sehec.utils import get_2D_ratemap, OnlineRateMap
from sehec.experiments.hafting_2008_data import Hafting2008Data


class Wernle20118Data(Hafting2008Data):
    """ Data wrapper for https://www.nature.com/articles/s41593-017-0036-6
    The data can be obtained from https://doi.org/10.11582/2017.00023
    """

    def __init__(self, data_path=None, recording_index=None, experiment_name="Wernle2018", verbose=False):
        super().__init__(data_path=data_path, recording_index=recording_index,
                         experiment_name=experiment_name, verbose=verbose)

    def _find_data_path(self, data_path):
        if data_path is None:
            self.data_path = os.path.join(sehec.__path__[0], "experiments/wernle_2018/")
        else:
            self.data_path = data_path

    def _load_data(self):
        """
        This method loads the recording data of the paper generating the following attributes.
        For further details refer to the readme of the original repo for the paper with the variable description.

        Returns
        -------
        self.ratemap : ndarray
            (128 x 2) where each element is a 100x100 ratemap from 10 different rats
            The first column are the reatmaps before merging the room, second column after merging the room.
        self.ratemap_dev : ndarray

        self.pos_A_B : ndarray
            Position of rats before merging rooms
            (19 x 1) where each element is a (T, 4) recording where T are the total number of samples in time
            at a 50Hz frequency, and column 1 is the time in seconds, column 2 is x_position, column 3 is y_position,
            column 4 is speed index (1 if speed is above 5cm/s)
        self.pos_AB : ndarray
            Position of rats after merging rooms
            (19 x 1) where each element is a (T, 4) recording where T are the total number of samples in time
            at a 50Hz frequency, and column 1 is the time in seconds, column 2 is x_position, column 3 is y_position,
            column 4 is speed index (1 if speed is above 5cm/s)
        """

        # Load ratemaps
        self.inner_path = "nn_Data+Code/data/"
        self.ratemap = sio.loadmat(os.path.join(self.data_path, self.inner_path, "Figures_1_2_3/ratemaps.mat"))
        self.ratemap = self.ratemap["ratemaps"]

        # The following is the recording progression of grid cell patterns for 19 different cells
        # over 10 different rats

        self.ratemap_dev = sio.loadmat(os.path.join(self.data_path, self.inner_path,
                                                    r"Figure 4/ratemapsDevelopment.mat"))["ratemapsDevelopment"]

        self.pos_A_B = sio.loadmat(os.path.join(self.data_path, self.inner_path,
                                                r"Figure 4/posA_B.mat"))["posA_B"]

        self.pos_AB = sio.loadmat(os.path.join(self.data_path, self.inner_path,
                                               r"Figure 4/posAB.mat"))["posAB"]
        self.spikes_AB = sio.loadmat(os.path.join(self.data_path, self.inner_path,
                                                  r"Figure 4/spkAB.mat"))["spkAB"]
        self.arena_limits = np.array([[-100, 100], [-100, 100]])
        print("debug")

    def _create_dataframe(self):
        self.list = []
        rec_index = 0

        for i in range(self.ratemap_dev.shape[0]):
            self.list.append({"rec_index": rec_index, "session": i, "recorded_vars": ["time", "posx", "posy", "speed_index", "spikes"], "before_merge": False})
            rec_index += 1
            self.list.append({"rec_index": rec_index, "session": i, "recorded_vars": ["time", "posx", "posy", "speed_index"], "before_merge": True})
            rec_index += 1

        for i in range(self.ratemap.shape[0]):
            self.list.append({"rec_index": rec_index, "session": i, "recorded_vars": "ratemap", "before_merge": True})
            rec_index += 1
            self.list.append({"rec_index": rec_index, "session": i, "recorded_vars": "ratemap", "before_merge": False})
            rec_index += 1

        self.recording_list = pd.DataFrame(self.list).set_index("rec_index")

    def get_recording_data(self, recording_index=None):
        if type(recording_index) is list or type(recording_index) is tuple:
            data_list = []
            for ind in recording_index:
                session_info = self.recording_list.iloc[ind]
                if type(session_info["recorded_vars"]) is list:
                    sess_index = session_info["session"]
                    if session_info["before_merge"]:
                        sess_data = {"time": self.pos_A_B[sess_index, 0][:, 0],
                                     "posx": self.pos_A_B[sess_index, 0][:, 1],
                                     "posy": self.pos_A_B[sess_index, 0][:, 2],
                                     "speed_index": self.pos_A_B[sess_index, 0][:, 4]}

                    else:
                        sess_data = {"time": self.pos_A_B[sess_index, 0][:, 0],
                                     "posx": self.pos_A_B[sess_index, 0][:, 1],
                                     "posy": self.pos_A_B[sess_index, 0][:, 2],
                                     "speed_index": self.pos_A_B[sess_index, 0][:, 4],
                                     "spikes": self.spikes_AB[sess_index, 0][:, 0]}
                    rev_vars = list(sess_data.keys()) + ["dev", ]
                    data_list.append([sess_data, rev_vars, {"sess_index": sess_index}])

                elif session_info["recorded_vars"] == "ratemap":
                    sess_index = session_info["session"]
                    rev_vars = ["ratemap",]
                    if session_info["before_merge"]:
                        data_list.append([{"ratemap": self.ratemap[sess_index, 0]}, rev_vars, {"sess_index": sess_index}])
                    else:
                        data_list.append([{"ratemap": self.ratemap[sess_index, 1]}, rev_vars, {"sess_index": sess_index}])

            return data_list

        else:
            session_info = self.recording_list.iloc[recording_index]
            if type(session_info["recorded_vars"]) is list:
                sess_index = session_info["session"]
                if session_info["before_merge"]:
                    sess_data = {"time": self.pos_A_B[sess_index, 0][:, 0],
                                 "posx": self.pos_A_B[sess_index, 0][:, 1],
                                 "posy": self.pos_A_B[sess_index, 0][:, 2],
                                 "speed_index": self.pos_A_B[sess_index, 0][:, 3]}

                else:
                    sess_data = {"time": self.pos_A_B[sess_index, 0][:, 0],
                                 "posx": self.pos_A_B[sess_index, 0][:, 1],
                                 "posy": self.pos_A_B[sess_index, 0][:, 2],
                                 "speed_index": self.pos_A_B[sess_index, 0][:, 3],
                                 "spikes": self.spikes_AB[sess_index, 0][:, 0]}
                rev_vars = list(sess_data.keys()) + ["dev", ]
                return sess_data, rev_vars, {"sess_index": sess_index}

            elif session_info["recorded_vars"] == "ratemap":
                sess_index = session_info["session"]
                if session_info["before_merge"]:
                    sess_data = {"ratemap": self.ratemap[sess_index, 0]}
                else:
                    sess_data = {"ratemap": self.ratemap[sess_index, 1]}
                rev_vars = "ratemap"
                return sess_data, rev_vars, {"sess_index": sess_index}

    def plot_recording_tetr(self, recording_index=None, save_path=None, ax=None, tetrode_id=None):
        if type(recording_index) is list or type(recording_index) is tuple:
            axis_list = []
            for ind in recording_index:
                ind_axis = self.plot_recording_tetr(ind, save_path=save_path, ax=ax, tetrode_id=None)
                axis_list.append(ind_axis)
            return axis_list
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(10, 8))

        session_data, rev_vars = self.get_recording_data(recording_index)

        arena_width = self.arena_limits[0, 1] - self.arena_limits[0, 0]
        arena_depth = self.arena_limits[1, 1] - self.arena_limits[1, 0]

        if type(rev_vars) is list and "spikes" in rev_vars:
            time_array, test_spikes, x, y = session_data["time"], session_data["spikes"], session_data["posx"], \
                                            session_data["posy"]

            scale_ratio = 2  # To discretize space
            h, binx, biny = get_2D_ratemap(time_array, test_spikes, x, y, x_size=int(arena_width/scale_ratio),
                                           y_size=int(arena_depth/scale_ratio), filter_result=True)
        elif type(rev_vars) is list and not "spikes" in rev_vars:
            warnings.warn("No spike data pre merging")
            return
        else:
            h = session_data["ratemap"]

        merged = self.recording_list.iloc[recording_index]["before_merge"]
        merged_mssg = "merged" if not merged else "before_merge"

        sess_index = self.recording_list.iloc[recording_index]["session"]
        self._make_tetrode_plot(h, ax, "sess_index_"+str(sess_index)+"_"+merged_mssg, save_path)

    def plot_trajectory(self, recording_index=None, save_path=None, ax=None, plot_every=20):
        if type(recording_index) is list or type(recording_index) is tuple:
            axis_list = []
            for ind in recording_index:
                ind_axis = self.plot_trajectory(ind, save_path=save_path, ax=ax, tetrode_id=None)
                axis_list.append(ind_axis)
            return axis_list

        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(10, 8))

        session_data, rev_vars = self.get_recording_data(recording_index)

        arena_width = self.arena_limits[0, 1] - self.arena_limits[0, 0]
        arena_depth = self.arena_limits[1, 1] - self.arena_limits[1, 0]

        if type(rev_vars) is list:
            time_array, x, y = session_data["time"], session_data["posx"], session_data["posy"]
        else:
            warnings.warn("This index does not have position data")
            return

        self._make_trajectory_plot(x, y, ax, plot_every=plot_every)

    def plot_cell_comparison(self, session_index):
        if type(session_index) is list or type(session_index) is tuple:
            n_cells = len(session_index)
        else:
            n_cells = 1
            session_index = [session_index, ]
        f, ax = plt.subplots(n_cells, 2, figsize=(4, 4*n_cells))
        for i in range(n_cells):
            ax[i, 0].imshow(self.ratemap[session_index[i], 0])
            ax[i, 1].imshow(self.ratemap[session_index[i], 1])
            ax[i, 0].axhline(y=50, color="white")
            ax[i, 1].axhline(y=50, color="white", linestyle="--")
        ax[0, 0].set_title("Before merging")
        ax[0, 1].set_title("After merging")
        return ax

    def plot_development(self, n_cells=3, time_interval=(1.0, 2.0), merged=False, skip_every=10):
        if merged:
            pos = self.pos_AB[:n_cells, 0]
        else:
            pos = self.pos_A_B[:n_cells, 0]

        f, ax = plt.subplots(n_cells, 1, figsize=(3, 4*n_cells))
        for cell in range(n_cells):
            pos_i = pos[cell]
            init_sample = int(pos_i.shape[0] * (time_interval[0]*60.0)/(np.amax(pos_i[:, 0])))
            finish_sample = int(pos_i.shape[0] * (time_interval[1]*60.0)/(np.amax(pos_i[:, 0])))
            n_samples = finish_sample - init_sample
            cmap = mpl.cm.get_cmap("plasma")
            norm = plt.Normalize(init_sample, finish_sample)
            aux_x, aux_y = [], []
            prev_x, prev_y = pos_i[init_sample, 1], pos_i[finish_sample, 2]
            for sample_i in range(init_sample, finish_sample, skip_every):
                x, y = pos_i[sample_i, 1], pos_i[sample_i, 2]
                aux_x.append(x)
                aux_y.append(y)
                ax[cell].plot((prev_x, x), (prev_y, y), "-", color=cmap(norm(sample_i)), alpha=0.4, lw=0.5)
                prev_x = x
                prev_y = y
            ax[cell].set_ylim((-100, 100))
            ax[cell].set_xlim((-100, 100))
            sc = ax[cell].scatter(aux_x, aux_y, c=np.arange(init_sample, finish_sample, skip_every),
                                  vmin=init_sample, vmax=finish_sample, cmap="plasma", alpha=0.4, s=2)
            ax[cell].axhline(y=0, color="black")
            ticks = np.linspace(init_sample, finish_sample, num=10)
            tickslabels = np.round(np.linspace(init_sample/50/60, finish_sample/50/60, num=10), 1)
            cbar = plt.colorbar(sc, ax=ax[cell], ticks=ticks)
            cbar.ax.set_yticklabels(tickslabels, fontsize=8)
            cbar.ax.set_ylabel("Time [min]", rotation=270, labelpad=12)
        return ax

    def get_recorded_session(self, recording_index=None):
        return [None, None, None]


if __name__ == "__main__":
    data = WernleData()
    data.show_data()
    # data.plot_recording_tetr(3)
    # data.plot_trajectory(3)
    data.plot_cell_comparison(session_index=(125, 126))