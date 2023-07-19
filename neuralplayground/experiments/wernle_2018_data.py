import os.path
import warnings
from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio

import neuralplayground
from neuralplayground.datasets import fetch_data_path
from neuralplayground.experiments.hafting_2008_data import Hafting2008Data
from neuralplayground.utils import get_2D_ratemap
from neuralplayground.plotting.plot_utils import make_plot_trajectories , make_plot_rate_map


class Wernle2018Data(Hafting2008Data):
    """Data class for https://www.nature.com/articles/s41593-017-0036-6
    The data can be obtained from https://doi.org/10.11582/2017.00023
    """

    def __init__(
        self,
        data_path: str = None,
        recording_index: int = None,
        experiment_name: str = "FullWernleData",
        verbose: bool = False,
    ):
        """Wernle2018Data init, just initializing parent class Hafting2008Data

        Parameters
        ----------
        data_path: str
            if None, fetch the data from the NeuralPlayground data repository,
            else load data from given path
        recording_index: int
            if None, load data from default recording index
        experiment_name: str
            string to identify object in case of multiple instances
        verbose:
            if True, it will print original readme and data structure when initializing this object
        """
        super().__init__(
            data_path=data_path,
            recording_index=recording_index,
            experiment_name=experiment_name,
            verbose=verbose,
        )

    def _find_data_path(self, data_path: str):
        """Fetch data from NeuralPlayground data repository 
        if no data path is supplied by the user"""
        if data_path is None:
            self.data_path = fetch_data_path("wernle_2018")
        else:
            self.data_path = data_path

    def set_animal_data(self, recording_index: int = 0, tolerance: float = 1e-10):
        """Set position and head direction to be used by the Arena Class later"""
        session_data, rev_vars, rat_info = self.get_recording_data(recording_index)

        # Some of the recorded data in the list do not include positional data
        if type(rev_vars) is list:
            _, x, y = (
                session_data["time"],
                session_data["posx"],
                session_data["posy"],
            )
        else:
            warnings.warn("This index does not have position data")
            return
        pass

        # Position from meters to cm
        self.position = np.stack([x, y], axis=1) * 100
        head_direction = np.diff(self.position, axis=0)
        # Compute head direction from position derivative
        head_direction = head_direction / np.sqrt(np.sum(head_direction**2, axis=1) + tolerance)[..., np.newaxis]
        self.head_direction = head_direction

    def _load_data(self):
        """
        This method loads the recording data of the paper generating the following attributes.
        For further details refer to the readme of the original repo for the paper with the variable description.

        Returns (set the following attributes)
        -------
        self.ratemap : ndarray
            (128 x 2) where each element is a 100x100 ratemap from 10 different rats
            The first column are the reatmaps before merging the room, second column after merging the room.
        self.ratemap_dev : ndarray (19 x 2)
            ORIGINAL DESCRIPTION FROM README: contains ratemaps for the first trial in AB (19 cells from 10 rats);
            rows correspond to single cells, column 1: ratemap A¦B, column 2: first trial ratemap AB
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
        self.best_recording_index = 100  # Nice session recording as default
        self.inner_path = "nn_Data+Code/data/"
        self.ratemap = sio.loadmat(os.path.join(self.data_path, self.inner_path, "Figures_1_2_3/ratemaps.mat"))
        self.ratemap = self.ratemap["ratemaps"]

        # The following is the recording progression of grid cell patterns for 19 different cells
        # over 10 different rats

        self.ratemap_dev = sio.loadmat(os.path.join(self.data_path, self.inner_path, r"Figure 4/ratemapsDevelopment.mat"))[
            "ratemapsDevelopment"
        ]

        self.pos_A_B = sio.loadmat(os.path.join(self.data_path, self.inner_path, r"Figure 4/posA_B.mat"))["posA_B"]

        self.pos_AB = sio.loadmat(os.path.join(self.data_path, self.inner_path, r"Figure 4/posAB.mat"))["posAB"]
        self.spikes_AB = sio.loadmat(os.path.join(self.data_path, self.inner_path, r"Figure 4/spkAB.mat"))["spkAB"]
        # Setting Arena Limits in cm
        self.arena_limits = np.array([[-100, 100], [-100, 100]])

    def _create_dataframe(self):
        """Generate dataframe for easy display and access of data"""
        self.list = []
        rec_index = 0

        # Two types of data available, ratemaps, and full trajectories with spikes
        for i in range(self.ratemap_dev.shape[0]):
            # full trajectories with spikes
            self.list.append(
                {
                    "rec_index": rec_index,
                    "session": i,
                    "recorded_vars": [
                        "time",
                        "posx",
                        "posy",
                        "speed_index",
                        "spikes",
                        "ratemap_dev",
                    ],
                    "before_merge": False,
                }
            )
            rec_index += 1
            self.list.append(
                {
                    "rec_index": rec_index,
                    "session": i,
                    "recorded_vars": [
                        "time",
                        "posx",
                        "posy",
                        "speed_index",
                        "ratemap_dev",
                    ],
                    "before_merge": True,
                }
            )
            rec_index += 1

        for i in range(self.ratemap.shape[0]):
            # Ratemaps before and after merge
            self.list.append(
                {
                    "rec_index": rec_index,
                    "session": i,
                    "recorded_vars": "ratemap",
                    "before_merge": True,
                }
            )
            rec_index += 1
            self.list.append(
                {
                    "rec_index": rec_index,
                    "session": i,
                    "recorded_vars": "ratemap",
                    "before_merge": False,
                }
            )
            rec_index += 1

        self.recording_list = pd.DataFrame(self.list).set_index("rec_index")

    def get_recording_data(self, recording_index: Union[int, list, tuple] = None):
        """Get experimental data for a given recordin index1

        Parameters
        ----------
        recording_index: int, list of ints or tuple of ints
            recording identifier, index in pandas dataframe with listed data

        Returns (If given a list of recording_index, it will return a list of following parameters)
        -------
        sess_data: dict
            Dictionary with recorded raw data from the session of the respective recording index
            Format of this data follows original readme from the authors of the experiments
        rec_vars: list of str
            keys of session_data dict, recorded variables for a given session
        identifiers: dict
            Dictionary with rat_id and session_id of the returned session data
        """
        if type(recording_index) is list or type(recording_index) is tuple:
            data_list = []
            for ind in recording_index:
                session_info = self.recording_list.iloc[ind]
                if type(session_info["recorded_vars"]) is list:
                    sess_index = session_info["session"]
                    if session_info["before_merge"]:
                        sess_data = {
                            "time": self.pos_A_B[sess_index, 0][:, 0],
                            "posx": self.pos_A_B[sess_index, 0][:, 1],
                            "posy": self.pos_A_B[sess_index, 0][:, 2],
                            "speed_index": self.pos_A_B[sess_index, 0][:, 3],
                            "ratemap_dev": self.ratemap_dev[sess_index, 0],
                        }
                    else:
                        sess_data = {
                            "time": self.pos_AB[sess_index, 0][:, 0],
                            "posx": self.pos_AB[sess_index, 0][:, 1],
                            "posy": self.pos_AB[sess_index, 0][:, 2],
                            "speed_index": self.pos_AB[sess_index, 0][:, 3],
                            "spikes": self.spikes_AB[sess_index, 0][:, 0],
                            "ratemap_dev": self.ratemap_dev[sess_index, 1],
                        }
                    rev_vars = list(sess_data.keys()) + [
                        "dev",
                    ]
                    identifiers = {"sess_index": sess_index}
                    data_list.append([sess_data, rev_vars, identifiers])

                elif session_info["recorded_vars"] == "ratemap":
                    sess_index = session_info["session"]
                    rev_vars = [
                        "ratemap",
                    ]
                    identifiers = {"sess_index": sess_index}
                    if session_info["before_merge"]:
                        data_list.append(
                            [
                                {"ratemap": self.ratemap[sess_index, 0]},
                                rev_vars,
                                identifiers,
                            ]
                        )
                    else:
                        data_list.append(
                            [
                                {"ratemap": self.ratemap[sess_index, 1]},
                                rev_vars,
                                identifiers,
                            ]
                        )

            return data_list

        else:
            if recording_index is None:
                recording_index = self.best_recording_index
            session_info = self.recording_list.iloc[recording_index]
            if type(session_info["recorded_vars"]) is list:
                sess_index = session_info["session"]
                identifiers = {"sess_index": sess_index}
                if session_info["before_merge"]:
                    sess_data = {
                        "time": self.pos_A_B[sess_index, 0][:, 0],
                        "posx": self.pos_A_B[sess_index, 0][:, 1],
                        "posy": self.pos_A_B[sess_index, 0][:, 2],
                        "speed_index": self.pos_A_B[sess_index, 0][:, 3],
                        "ratemap_dev": self.ratemap_dev[sess_index, 0],
                    }
                else:
                    sess_data = {
                        "time": self.pos_AB[sess_index, 0][:, 0],
                        "posx": self.pos_AB[sess_index, 0][:, 1],
                        "posy": self.pos_AB[sess_index, 0][:, 2],
                        "speed_index": self.pos_AB[sess_index, 0][:, 3],
                        "spikes": self.spikes_AB[sess_index, 0][:, 0],
                        "ratemap_dev": self.ratemap_dev[sess_index, 1],
                    }
                rev_vars = list(sess_data.keys()) + [
                    "dev",
                ]
                return sess_data, rev_vars, identifiers

            elif session_info["recorded_vars"] == "ratemap":
                sess_index = session_info["session"]
                identifiers = {"sess_index": sess_index}
                if session_info["before_merge"]:
                    sess_data = {"ratemap": self.ratemap[sess_index, 0]}
                else:
                    sess_data = {"ratemap": self.ratemap[sess_index, 1]}
                rev_vars = "ratemap"
                return sess_data, rev_vars, identifiers

    def plot_recording_tetr(
        self,
        recording_index: Union[int, tuple, list] = None,
        save_path: Union[str, tuple, list] = None,
        ax: Union[mpl.axes.Axes, tuple, list] = None,
        tetrode_id: Union[str, tuple, list] = None,
        bin_size: float = 2.0,
    ):
        """Plot tetrode ratemap from spike data for a given recording index or a list of recording index.
        If given a list or tuple as argument, all arguments must be list, tuple, or None.

        Parameters
        ----------
        recording_index: int, tuple of ints, list of ints
            recording index to plot spike ratemap, if list or tuple, it will recursively call this function
            to make a plot per recording index. If this argument is list or tuple, the rest of variables must
            be list or tuple with their respective types, or keep the default None value.
        save_path: str, list of str, tuple of str
            saving path of the generated figure, if None, no figure is saved
        ax: mpl.axes._subplots.AxesSubplot (matplotlib axis from subplots), list of ax, or tuple of ax
            axis or list of axis from subplot from matplotlib where the ratemap will be plotted.
            If None, ax is generated with default options.
        tetrode_id: str, list of str, or tuple of str
            tetrode id in the corresponding session
        bin_size: float
            bin size to discretize space when computing ratemap

        Returns
        -------
        h: ndarray (nybins, nxbins)
            Number of spikes falling on each bin through the recorded session, nybins number of bins in y axis,
            nxbins number of bins in x axis
        binx: ndarray (nxbins +1,)
            bin limits of the ratemap on the x axis
        biny: ndarray (nybins +1,)
            bin limits of the ratemap on the y axis
        (when using list pr tuple as argument, this function return a list or tuple of the variables listed above)
        """
        # Recursive call of this function in case of list or tuple
        if recording_index is None:
            recording_index = self.best_recording_index

        if type(recording_index) is list or type(recording_index) is tuple:
            axis_list = []
            for i, ind in enumerate(recording_index):
                # Checking if rest of variables are default or list values
                if save_path is not None:
                    save_path_i = save_path[i]
                else:
                    save_path_i = None
                if ax is not None:
                    ax_i = ax[0]
                else:
                    ax_i = None
                if tetrode_id is not None:
                    tetrode_id_i = tetrode_id[i]
                else:
                    tetrode_id_i = None
                ind_axis = self.plot_recording_tetr(ind, save_path=save_path_i, ax=ax_i, tetrode_id=tetrode_id_i)
                axis_list.append(ind_axis)
            return axis_list

        # Generate axis in case ax is None
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Recall recorded data

        h, binx, biny = self.recording_tetr()

        # Adding merging status to plot title

        merged = self.recording_list.iloc[recording_index]["before_merge"]
        merged_mssg = "merged" if not merged else "before_merge"

        sess_index = self.recording_list.iloc[recording_index]["session"]
        # Use auxiliary function to make the plot
        ax=make_plot_rate_map(h, ax, "sess_index_" + str(sess_index) + "_" + merged_mssg,"width","depth","Firing rate")
        # Save if save_path is not None
        if save_path is None:
            pass
        else:
            plt.savefig(save_path, bbox_inches="tight")
        # Return ratemap values, x bin limits and y bin limits
        return ax

    def recording_tetr(self, recording_index: Union[int, tuple, list] = None,
                            save_path: Union[str, tuple, list] = None,
                            tetrode_id: Union[str, tuple, list] = None,
                            bin_size: float = 2.0):


        # Recall recorded data
        session_data, rev_vars, rat_info = self.get_recording_data(recording_index)

        arena_width = self.arena_limits[0, 1] - self.arena_limits[0, 0]
        arena_depth = self.arena_limits[1, 1] - self.arena_limits[1, 0]

        # Check of the session has spikes to compute ratemap
        # Plot pre-computed ratemap otherwise
        if type(rev_vars) is list and "spikes" in rev_vars:
            time_array, test_spikes, x, y = (
                session_data["time"],
                session_data["spikes"],
                session_data["posx"],
                session_data["posy"],
            )

            h, binx, biny = get_2D_ratemap(
                time_array,
                test_spikes,
                x,
                y,
                x_size=int(arena_width / bin_size),
                y_size=int(arena_depth / bin_size),
                filter_result=True,
            )
        elif type(rev_vars) is list and "spikes" not in rev_vars:
            warnings.warn("No spike data pre merging")
            return
        else:
            h = session_data["ratemap"]
            binx = np.linspace(self.arena_limits[0, 0], self.arena_limits[0, 1], num=h.shape[1])
            biny = np.linspace(self.arena_limits[1, 0], self.arena_limits[1, 1], num=h.shape[0])

        return h, binx, biny


    def plot_trajectory(
        self,
        recording_index: Union[int, tuple, list] = None,
        save_path: Union[str, tuple, list] = None,
        ax: Union[mpl.axes.Axes, tuple, list] = None,
        plot_every: int = 20,
    ):
        """Plot animal trajectory from a given recording index, corresponding to a recording session

        Parameters
        ----------
        recording_index: int, tuple of ints, list of ints
            recording index to plot spike ratemap, if list or tuple, it will recursively call this function
            to make a plot per recording index. If this argument is list or tuple, the rest of variables must
            be list or tuple with their respective types, or keep the default None value.
        save_path: str, list of str, tuple of str
            saving path of the generated figure, if None, no figure is saved
        ax: mpl.axes._subplots.AxesSubplot (matplotlib axis from subplots), list of ax, or tuple of ax
            axis or list of axis from subplot from matplotlib where the ratemap will be plotted.
            If None, ax is generated with default options.
        plot_every: int
            time steps skipped to make the plot to reduce cluttering

        Returns
        -------
        x: ndarray (n_samples,)
            x position throughout recording of the given session
        y: ndarray (n_samples,)
            y position throughout recording of the given session
        time_array: ndarray (n_samples,)
            array with the timestamps in seconds per position of the given session

        """
        if recording_index is None:
            recording_index = self.best_recording_index

        if type(recording_index) is list or type(recording_index) is tuple:
            axis_list = []
            for i, ind in enumerate(recording_index):
                # Checking if rest of variables are default or list values
                if save_path is not None:
                    save_path_i = save_path[i]
                else:
                    save_path_i = None
                if ax is not None:
                    ax_i = ax[0]
                else:
                    ax_i = None
                ind_axis = self.plot_trajectory(ind, save_path=save_path_i, ax=ax_i, plot_every=plot_every)
                axis_list.append(ind_axis)
            return axis_list

        # Generate axis in case ax is None
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(10, 8))

        session_data, rev_vars, identifiers = self.get_recording_data(recording_index)

        self.arena_limits[0, 1] - self.arena_limits[0, 0]
        self.arena_limits[1, 1] - self.arena_limits[1, 0]

        if type(rev_vars) is list:
            time_array, x, y = (
                session_data["time"],
                session_data["posx"],
                session_data["posy"],
            )
        else:
            warnings.warn("This index does not have position data")
            return

        # Helper function to format the trajectory plot

        ax =  make_plot_trajectories(self.arena_limits, x, y, ax, plot_every)
        if save_path is None:
            pass
        else:
            plt.savefig(save_path, bbox_inches="tight")
        return x, y, time_array

    def plot_merging_comparison(self, session_index: Union[int, list, tuple]):
        """Plot ratemaps before and after merging for a given session index

        Parameters
        ----------
        session_index: int, list of ints, tuple of ints
            session id to identify ratemap from dataset
        Returns
        -------
        ratemaps_before: list
            List of ratemaps before the merging
        ratemaps_after: list
            List of ratempas after the merging
        ax: mpl.axes._subplots.AxesSubplot (matplotlib axis from subplots)
            plot axes from matplotlib for further formating
        """
        if type(session_index) is list or type(session_index) is tuple:
            n_cells = len(session_index)
        else:
            n_cells = 1
            session_index = [
                session_index,
            ]
        f, ax = plt.subplots(n_cells, 2, figsize=(8, 5 * n_cells))
        ratemaps_before = []
        ratemaps_after = []
        for i in range(n_cells):
            make_plot_rate_map(self.ratemap[session_index[i], 0],  ax[i, 0], "Before merging", "width", "depth","Firing rate")
            make_plot_rate_map(self.ratemap[session_index[i], 1], ax[i, 1], "After merging", "width", "depth", "Firing rate")
            ratemaps_before.append(self.ratemap[session_index[i], 0])
            ratemaps_after.append(self.ratemap[session_index[i], 1])
            ax[i, 0].axhline(y=50, color="white")
            ax[i, 1].axhline(y=50, color="white", linestyle="--")
        return ratemaps_before, ratemaps_after, ax



    def get_recorded_session(self, recording_index=None):
        # Not used, override to avoid issues
        return [None, None, None]


if __name__ == "__main__":
    data = Wernle2018Data()
    data.show_data()
    data.plot_cell_comparison(session_index=(125, 126))
