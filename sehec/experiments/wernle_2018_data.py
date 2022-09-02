import os.path

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import sehec
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.interpolate import interp1d
from sehec.utils import get_2D_ratemap, OnlineRateMap


class WernleData(object):
    """ Data wrapper for https://www.nature.com/articles/s41593-017-0036-6
    The data can be obtained from https://doi.org/10.11582/2017.00023
    """

    def __init__(self, data_path=None, experiment_name="MergingRoomData", verbose=False):
        self.experiment_name = experiment_name
        if data_path==None:
            self.data_path = os.path.join(sehec.__path__[0], "experiments/wernle_2018/")
        else:
            self.data_path = data_path
        self.inner_path = "nn_Data+Code/data/"
        self.cmap = "jet"
        self._load_data()

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
        self.ratemap = sio.loadmat(os.path.join(self.data_path, self.inner_path, "Figures_1_2_3/ratemaps.mat"))
        self.ratemap = self.ratemap["ratemaps"]

        # The following is the recording progression of grid cell patterns for 19 different cells
        # over 10 different rats

        self.ratemap_dev = sio.loadmat(os.path.join(self.data_path, self.inner_path,
                                                    r"Figure 4/ratemapsDevelopment.mat"))["ratemapsDevelopment"]

        self.pos_A_B = sio.loadmat(os.path.join(self.data_path, self.inner_path,
                                                r"Figure 4/posA_B.mat"))["posA_B"]
        singlerun = self.pos_A_B[0, 0]
        print(singlerun)
        for i, pos in enumerate(singlerun):
            if pos[2] > 0:
                if singlerun[i+1000, 2] > 0:
                    print("Switch room at: ", pos)
                    break

        self.pos_AB = sio.loadmat(os.path.join(self.data_path, self.inner_path,
                                               r"Figure 4/posAB.mat"))["posAB"]
        self.spikes_AB = sio.loadmat(os.path.join(self.data_path, self.inner_path,
                                                  r"Figure 4/spkAB.mat"))["spkAB"]

    def plot_cell_comparison(self, n_cells=5):
        f, ax = plt.subplots(n_cells, 2, figsize=(4, 4*n_cells))
        for i in range(n_cells):
            ax[i, 0].imshow(self.ratemap[i, 0])
            ax[i, 1].imshow(self.ratemap[i, 1])
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


