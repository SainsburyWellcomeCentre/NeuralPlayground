import os.path

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.interpolate import interp1d


class WernleData(object):
    """ Data wrapper for https://www.nature.com/articles/s41593-017-0036-6
    The data can be obtained from https://doi.org/10.11582/2017.00023
    """

    def __init__(self, data_path, experiment_name="MergingRoomData", verbose=False):
        self.experiment_name = experiment_name
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

        #
        self.ratemap_dev = sio.loadmat(os.path.join(self.data_path, self.inner_path,
                                                    r"Figure 4/ratemapsDevelopment.mat"))["ratemapsDevelopment"]


        self.pos_A_B = sio.loadmat(os.path.join(self.data_path, self.inner_path,
                                                r"Figure 4/posA_B.mat"))["posA_B"]
        singlerun = self.pos_A_B[0,0]
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


class OnlineRateMap(object):

    def __init__(self, spikes, position, size=(100, 100), x_range=(-100, 100), y_range=(-100, 100)):
        """
        Creates a ratemap where it is possible to distinguish firing rate and traversed position at the same time

        Parameters
        ----------
        spikes : ndarray
            (n_spikes,) shaped array with times of spikes for a single cell
        position : ndarray
            (timestamps, 3) shaped array with timestamp time in seconds,
             position in x-dim on column 2 and y-dim on column 2 for each timestamp
        size : tuple
            (2,) shaped tupple with the size of the desired ratemap
        x_range : tuple
            (2,) shaped tuple with limits on x-dim
        y_range
            (2,) shaped tuple with limits on y-dim
        """
        self.ratemap = np.empty(shape=size)
        self.ratemap[:] = np.nan
        self.size = size
        self.x_range = x_range
        self.y_range = y_range
        self.x_pos_bins = np.linspace(x_range[0], x_range[1], num=size[0]+1)
        self.y_pos_bins = np.linspace(y_range[0], y_range[1], num=size[1]+1)
        self.spikes = spikes
        self.position = position
        self.last_t_end = 0
        self.last_t_init = 0

    def get_ratemap(self, t_end, t_init=0, interp_factor=10):
        """
        Computes the ratemap using given spike train (self.spikes) and position (self.position)

        Parameters
        ----------
        t : float
            maximum time of recording to consider for the ratemap
        t_init : float
            starting time of recording to consider for the ratemap, default set to 0
        Returns
        -------
        ratemap : ndarray
            updated ratemap for times between t_init and t
        """

        # Get spikes and position within the range
        pos = self.position[np.logical_and(self.position[:, 0] >= t_init, self.position[:, 0] < t_end), :]

        # Interpolate position for smooth plot
        if interp_factor != 1:
            t = pos[:, 0]
            x = pos[:, 1]
            y = pos[:, 2]
            t_inter = np.linspace(np.amin(t), np.amax(t), num=len(t)*interp_factor, endpoint=False)
            fx = interp1d(t, x)
            fy = interp1d(t, y)
            x_inter = fx(t_inter)
            y_inter = fy(t_inter)
            pos = np.stack([t_inter, x_inter, y_inter], axis=1)

        spk = self.spikes[np.logical_and(self.spikes >= t_init, self.spikes < t_end)]

        h_spk, bin_spk = np.histogram(spk, bins=np.linspace(t_init, t_end, num=pos.shape[0]+1))

        # Convert position to indexes in the ratemap
        x_index = (pos[:, 1] - self.x_range[0])/(self.x_range[1]-self.x_range[0])*self.size[1]
        y_index = (pos[:, 2] - self.y_range[0])/(self.y_range[1]-self.y_range[0])*self.size[0]
        x_index, y_index = x_index.astype(int), y_index.astype(int)

        # update ratemap pixels
        for pos_i in range(len(x_index)):
            t = pos[pos_i, 0]
            ratemap_val = self.ratemap[y_index[pos_i], x_index[pos_i]]
            if np.isnan(ratemap_val):
                self.ratemap[y_index[pos_i], x_index[pos_i]] = 0
            self.ratemap[y_index[pos_i], x_index[pos_i]] += h_spk[pos_i]

        smooth_ratemap = self.get_smooth_ratemap()

        self.last_t_init = t_init
        self.last_t_end = t_end
        return smooth_ratemap

    def update_ratemap(self, dt, interp_factor=1):
        t_init = self.last_t_init
        t_end = self.last_t_end + dt
        pos = self.position[np.logical_and(self.position[:, 0] >= t_init, self.position[:, 0] < t_end), :]

        # Interpolate position for smooth plot
        if interp_factor != 1:
            t = pos[:, 0]
            x = pos[:, 1]
            y = pos[:, 2]
            t_inter = np.linspace(np.amin(t), np.amax(t), num=len(t)*interp_factor, endpoint=False)
            fx = interp1d(t, x)
            fy = interp1d(t, y)
            x_inter = fx(t_inter)
            y_inter = fy(t_inter)
            pos = np.stack([t_inter, x_inter, y_inter], axis=1)

        spk = self.spikes[np.logical_and(self.spikes >= t_init, self.spikes < t_end)]

        h_spk, bin_spk = np.histogram(spk, bins=np.linspace(t_init, t_end, num=pos.shape[0]+1))

        # Convert position to indexes in the ratemap
        x_index = (pos[:, 1] - self.x_range[0])/(self.x_range[1]-self.x_range[0])*self.size[1]
        y_index = (pos[:, 2] - self.y_range[0])/(self.y_range[1]-self.y_range[0])*self.size[0]
        x_index, y_index = x_index.astype(int), y_index.astype(int)

        # update ratemap pixels
        for pos_i in range(len(x_index)):
            t = pos[pos_i, 0]
            ratemap_val = self.ratemap[y_index[pos_i], x_index[pos_i]]
            if np.isnan(ratemap_val):
                self.ratemap[y_index[pos_i], x_index[pos_i]] = 0
            self.ratemap[y_index[pos_i], x_index[pos_i]] += h_spk[pos_i]

        smooth_ratemap = self.get_smooth_ratemap()

        self.last_t_init = t_init
        self.last_t_end = t_end
        return smooth_ratemap

    def get_smooth_ratemap(self):
        nan_indexes = np.isnan(self.ratemap)
        aux_ratemap = np.copy(self.ratemap)
        aux_ratemap[nan_indexes] = 0
        filtered_ratemap = gaussian_filter(aux_ratemap, 3.5)
        filtered_ratemap[nan_indexes] = np.nan
        return filtered_ratemap



if __name__ == "__main__":
    data_path = "Wernle2018/"
    data = WernleData(data_path=data_path, verbose=True)
    data.plot_development(n_cells=4, time_interval=(0, 5), skip_every=10)
    data.plot_cell_comparison()
    plt.show()

    print("debug")

    f, ax = plt.subplots(4, 6, figsize=(4*6, 4*4))
    for j in range(4):
        spikes = data.spikes_AB[j+5, 0][:, 0]
        pos = data.pos_AB[j+5, 0]
        ratemap = OnlineRateMap(spikes=spikes, position=pos, size=(100, 100))
        for i in range(6):
            rm = ratemap.get_ratemap(t_end=((i+1)*5)*60, t_init=0, interp_factor=1)
            ax[j, i].imshow(rm, cmap="jet")

    f, ax = plt.subplots(4, 6, figsize=(4*6, 4*4))
    for j in range(4):
        spikes = data.spikes_AB[j+5, 0][:, 0]
        pos = data.pos_AB[j+5, 0]
        ratemap = OnlineRateMap(spikes=spikes, position=pos, size=(100, 100))
        for i in range(6):
            rm = ratemap.update_ratemap(dt=5*60, interp_factor=1)
            ax[j, i].imshow(rm, cmap="jet")

    plt.show()
