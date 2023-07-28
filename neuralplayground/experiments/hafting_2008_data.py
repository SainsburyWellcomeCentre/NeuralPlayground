import glob
import os.path
from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from IPython.display import display

from neuralplayground.datasets import fetch_data_path
from neuralplayground.utils import clean_data, get_2D_ratemap
from neuralplayground.plotting.plot_utils import make_plot_trajectories , make_plot_rate_map


from .experiment_core import Experiment


class Hafting2008Data(Experiment):
    """Data class for Hafting et al. 2008. https://www.nature.com/articles/nature06957
    The data can be obtained from https://archive.norstore.no/pages/public/datasetDetail.jsf?id=C43035A4-5CC5-44F2-B207-126922523FD9
    This class only consider animal raw animal trajectories and neural recordings
    This class is also used for Sargolini2006Data due to its similar data structure
    """

    def __init__(
        self,
        data_path: str = None,
        recording_index: int = None,
        experiment_name: str = "FullHaftingData",
        verbose: bool = False,
    ):
        """Hafting2008Data Init

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
        self.experiment_name = experiment_name
        self._find_data_path(data_path)
        self._load_data()
        self._create_dataframe()
        self.rat_id, self.sess, self.rec_vars = self.get_recorded_session(recording_index)
        self.set_animal_data()
        if verbose:
            self.show_readme()
            self.show_data()

    def set_animal_data(self, recording_index: int = 0, tolerance: float = 1e-10):
        """Set position and head direction to be used by the Arena Class later"""
        session_data, rev_vars, rat_info = self.get_recording_data(recording_index)
        tetrode_id = self._find_tetrode(rev_vars)
        time_array, test_spikes, x, y = self.get_tetrode_data(session_data, tetrode_id)

        self.position = np.stack([x, y], axis=1)
        head_direction = np.diff(self.position, axis=0)
        # Compute head direction from position derivative
        head_direction = head_direction / np.sqrt(np.sum(head_direction**2, axis=1) + tolerance)[..., np.newaxis]
        self.head_direction = head_direction

    def _find_data_path(self, data_path: str):
        """Fetch data from NeuralPlayground data repository 
        if no data path is supplied by the user"""
        if data_path is None:
            self.data_path = fetch_data_path("hafting_2008")
        else:
            self.data_path = data_path

    def _load_data(self):
        """Parse data according to specific data format
        if you are a user check the notebook examples"""
        self.best_recording_index = 4  # Nice session recording as default
        # Arena limits from the experimental setting, first row x limits, second row y limits, in cm
        self.arena_limits = np.array([[-200, 200], [-20, 20]])

        data_path_list = glob.glob(self.data_path + "*.mat")
        mice_ids = np.unique([dp.split("/")[-1][:5] for dp in data_path_list])
        # Initialize data dictionary, later handled by this object itself (so don't worry about this)
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
                    # Interpolate to replace NaNs and stuff
                    cleaned_data = clean_data(sio.loadmat(r_path[0]))
                    self.data_per_animal[m_id][sess][session_info] = cleaned_data

    def _create_dataframe(self):
        """Generate dataframe for easy display and access of data"""
        self.list = []
        idx = 0
        for rat_id, rat_sess in self.data_per_animal.items():
            for sess, recorded_vars in rat_sess.items():
                self.list.append(
                    {
                        "rec_index": idx,
                        "rat_id": rat_id,
                        "session": sess,
                        "recorded_vars": list(recorded_vars.keys()),
                    }
                )
                idx += 1
        self.recording_list = pd.DataFrame(self.list).set_index("rec_index")

    def show_data(self, full_dataframe: bool = False):
        """Print of available data recorded in the experiment

        Parameters
        ----------
        full_dataframe: bool
            if True, it will show all available data, a small sample otherwise

        Returns
        -------
        recording_list: Pandas dataframe
            List of available data, columns with rat_id, recording session and recorded variables
        """
        print("Dataframe with recordings")
        if full_dataframe:
            pd.set_option("display.max_rows", None)
            pd.set_option("display.max_columns", None)
        display(self.recording_list)
        return self.recording_list

    def show_readme(self):
        """Print original readme of the dataset"""
        readme_path = glob.glob(self.data_path + "readme" + "*.txt")[0]
        with open(readme_path, "r") as fin:
            print(fin.read())

    def get_recorded_session(self, recording_index: int = None):
        """Get identifiers to sort the experimental data

        Parameters
        ----------
        recording_index: int
            recording identifier, index in pandas dataframe with listed data

        Returns
        -------
        rat_id: str
            rat identifier from experiment
        sess: str
            recording session identifier from experiment
        recorded_vars: list of str
            Variables recorded from a given session
        """
        if recording_index is None:
            recording_index = self.best_recording_index
        list_item = self.recording_list.iloc[recording_index]
        rat_id, sess, recorded_vars = (
            list_item["rat_id"],
            list_item["session"],
            list_item["recorded_vars"],
        )
        return rat_id, sess, recorded_vars

    def get_recording_data(self, recording_index: int = None):
        """Get experimental data for a given recordin index

        Parameters
        ----------
        recording_index: int
            recording identifier, index in pandas dataframe with listed data

        Returns
        -------
        session_data: dict
            Dictionary with recorded raw data from the session of the respective recording index
            Format of this data follows original readme from the authors of the experiments
        rec_vars: list of str
            keys of session_data dict, recorded variables for a given session
        identifiers: dict
            Dictionary with rat_id and session_id of the returned session data
        """
        if recording_index is None:
            recording_index = self.best_recording_index
        if type(recording_index) is list or type(recording_index) is tuple:
            data_list = []
            for ind in recording_index:
                rat_id, sess, rec_vars = self.get_recorded_session(ind)
                session_data = self.data_per_animal[rat_id][sess]
                data_list.append([session_data, rec_vars, {"rat_id": rat_id, "session": sess}])
            return data_list
        else:
            rat_id, sess, rec_vars = self.get_recorded_session(recording_index)
            session_data = self.data_per_animal[rat_id][sess]
            identifiers = {"rat_id": rat_id, "sess": sess}
            return session_data, rec_vars, identifiers

    def _find_tetrode(self, rev_vars: list):
        """Static function to find tetrode id in a multiple tetrode recording session

        Parameters
        ----------
        rev_vars: list of strings
            recorded variables for a given session, tetrode id within it

        Returns
        -------
        tetrode_id: str
            found first tetrode id in the recorded variable list
        """
        tetrode_id = next(
            var_name for var_name in rev_vars if (var_name != "position") and (("t" in var_name) or ("T" in var_name))
        )
        return tetrode_id

    def get_tetrode_data(self, session_data: str = None, tetrode_id: str = None):
        """Return time stamp, position and spikes for a given session and tetrode

        Parameters
        ----------
        session_data: str
            if None, the session used corresponds to the default recording index
        tetrode_id:
            tetrode id in the corresponding session

        Returns
        -------
        time_array: ndarray (n_samples,)
            array with the timestamps in seconds per position of the given session
        test_spikes: ndarray (n_spikes,)
            spike times in seconds of the given session
        x: ndarray (n_samples,)
            x position throughout recording of the given session
        y: ndarray (n_samples,)
            y position throughout recording of the given session
        """
        # Use default recording index when session is not given
        if session_data is None:
            session_data, rev_vars, rat_info = self.get_recording_data(recording_index=self.best_recording_index)
            tetrode_id = self._find_tetrode(rev_vars)

        position_data = session_data["position"]
        x1, y1 = position_data["posx"][:, 0], position_data["posy"][:, 0]
        x2, y2 = x1, y1
        # Selecting positional data
        x = np.clip(x2, a_min=self.arena_limits[0, 0], a_max=self.arena_limits[0, 1])
        y = np.clip(y2, a_min=self.arena_limits[1, 0], a_max=self.arena_limits[1, 1])
        time_array = position_data["post"][:]
        tetrode_data = session_data[tetrode_id]
        test_spikes = tetrode_data["ts"][:,]
        test_spikes = test_spikes[:, 0]
        time_array = time_array[:, 0]

        return time_array, test_spikes, x, y


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
        # Compute ratemap matrices from data
        session_data, rev_vars, rat_info = self.get_recording_data(recording_index)
        if tetrode_id is None:
            tetrode_id = self._find_tetrode(rev_vars)


        h, binx, biny = self.recording_tetr(recording_index, save_path, tetrode_id, bin_size)

        # Use auxiliary function to make the plot
        ax = make_plot_rate_map(h, ax, 'rat: '+str(rat_info['rat_id'])+' sess: '+str(rat_info['sess'])+' tetrode: '+tetrode_id,"width","depth","Firing rate")
        if save_path is None:
            return h, binx, biny
        else:
            plt.savefig(save_path, bbox_inches="tight")
            # Return ratemap values, x bin limits and y bin limits
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
            f, ax = plt.subplots(1, 1, figsize=(8, 6))

        session_data, rev_vars, rat_info = self.get_recording_data(recording_index)
        tetrode_id = self._find_tetrode(rev_vars)

        time_array, test_spikes, x, y = self.get_tetrode_data(session_data, tetrode_id)
        # Helper function to format the trajectory plot

        ax = make_plot_trajectories(self.arena_limits, x, y, ax, plot_every)

        # Save if save_path is not None
        if save_path is None:
            pass
        else:
            plt.savefig(save_path, bbox_inches="tight")
        return x, y, time_array


    def recording_tetr(self, recording_index: Union[int, tuple, list] = None,
                            save_path: Union[str, tuple, list] = None,
                            tetrode_id: Union[str, tuple, list] = None,
                            bin_size: float = 2.0):
        """ tetrode ratemap from spike data for a given recording index or a list of recording index.
        If given a list or tuple as argument, all arguments must be list, tuple, or None.

        Parameters
        ----------
        recording_index: int, tuple of ints, list of ints
            recording index to plot spike ratemap, if list or tuple, it will recursively call this function
            to make a plot per recording index. If this argument is list or tuple, the rest of variables must
            be list or tuple with their respective types, or keep the default None value.
        save_path: str, list of str, tuple of str
            saving path of the generated figure, if None, no figure is saved
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
        session_data, rev_vars, rat_info = self.get_recording_data(recording_index)
        if tetrode_id is None:
            tetrode_id = self._find_tetrode(rev_vars)

        arena_width = self.arena_limits[0, 1] - self.arena_limits[0, 0]
        arena_depth = self.arena_limits[1, 1] - self.arena_limits[1, 0]

        # Recall spike data
        time_array, test_spikes, x, y = self.get_tetrode_data(session_data, tetrode_id)

        # Compute ratemap matrices from data
        h, binx, biny = get_2D_ratemap(time_array, test_spikes, x, y, x_size=int(arena_width / bin_size),
                                       y_size=int(arena_depth / bin_size), filter_result=True)

        return h, binx, biny
