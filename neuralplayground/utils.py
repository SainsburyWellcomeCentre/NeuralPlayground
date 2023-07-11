import json
import os
from datetime import datetime

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter


def check_crossing_wall(
    pre_state: np.ndarray,
    new_state: np.ndarray,
    wall: np.ndarray,
    wall_closenes: float = 1e-5,
    tolerance: float = 1e-9,
):
    """

    Parameters
    ----------
    pre_state : (2,) 2d-ndarray
        2d position of pre-movement
    new_state : (2,) 2d-ndarray
        2d position of post-movement
    wall : (2, 2) ndarray
        [[x1, y1], [x2, y2]] where (x1, y1) is on limit of the wall, (x2, y2) second limit of the wall
    wall_closenes : float
        how close the agent is allowed to be from the wall
    tolerance: float
        Small constant to avoid inverting a singular matrix

    Returns
    -------
    new_state: (2,) 2d-ndarray
        corrected new state. If it is not crossing the wall, then the new_state stays the same, if the state cross the
        wall, new_state will be corrected to a valid place without crossing the wall
    cross_wall: bool
        True if the change in state cross a wall
    """

    # Check if the line of the wall and the line between the states cross
    A = np.stack([np.diff(wall, axis=0)[0, :], -new_state + pre_state], axis=1)
    b = pre_state - wall[0, :]
    try:
        intersection = np.linalg.inv(A) @ b
    except Exception:
        intersection = np.linalg.inv(A + np.identity(A.shape[0]) * tolerance) @ b
    smaller_than_one = intersection <= 1
    larger_than_zero = intersection >= 0

    # If condition is true, then the points cross the wall
    cross_wall = np.alltrue(np.logical_and(smaller_than_one, larger_than_zero))
    if cross_wall:
        new_state = (intersection[-1] - wall_closenes) * (new_state - pre_state) + pre_state

    return new_state, cross_wall


def create_circular_wall(center: np.ndarray, radius: float, n_walls: int = 100):
    """Generate a circular wall by discretizing the circle into many walls.

    Parameters
    ----------
    center: ndarray (2,)
        Center of the circular wall
    radius: float
        Radius of the circular wall
    n_walls: int
        Number of walls used to discretize the circle

    Returns
    -------
    list_of_segments: list of walls
        n_walls that creates a circle
    """

    d_angle = 2 * np.pi / n_walls
    list_of_segments = []
    for i in range(n_walls):
        init_point = np.array([radius * np.cos(d_angle * i), radius * np.sin(d_angle * i)]) + center
        end_point = np.array([radius * np.cos(d_angle * (i + 1)), radius * np.sin(d_angle * (i + 1))]) + center
        wall = np.stack([init_point, end_point])
        list_of_segments.append(wall)
    return list_of_segments


def inheritors(klass):
    """Given a class, returns child classes (for future implementation of comparison board)"""
    subclasses = set()
    work = [klass]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return subclasses


def clean_data(data, keep_headers=False):
    """For a dictionary with positional data, this function replace nans by interpolated positions
    Used in data readers in the experiment classes

    Parameters
    ----------
    data: dict
        dictionary with values that might contain arrays with nans
    keep_headers: bool
        it will remove unwanted keys from the returned dictionary that are residuals of
        loading .mat files
        (sounds a bit cryptic, please check _load_data in experiment classes)

    Returns
    -------
    aux_dict: dict
        Modified dictionary with some keys removed and nans interpolated
    """
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
                f = interp1d(clean_x, clean_val, kind="cubic", fill_value="extrapolate")
                aux_dict[key] = f(x_range)[..., np.newaxis]
    return aux_dict


def get_2D_ratemap(
    time_array: np.ndarray,
    spikes: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x_size: float = 50,
    y_size: float = 50,
    filter_result: bool = False,
):
    """
    Parameters
    ----------
    time_array: ndarray (n_samples,)
        array with the timestamps in seconds per position of the given session
    spikes: ndarray (n_spikes,)
        spike times in seconds of the given session
    x: ndarray (n_samples,)
        x position throughout recording of the given session
    y: ndarray (n_samples,)
        y position throughout recording of the given session
    x_size: float
        bin size in x dim
    y_size: float
        bin size in y dim
    filter_result: bool
        If true, it will smooth the ratemap using a gaussian filter

    Returns
    -------
    h: ndarray (nybins, nxbins)
        Number of spikes falling on each bin through the recorded session, nybins number of bins in y axis,
        nxbins number of bins in x axis
    binx: ndarray (nxbins +1,)
        bin limits of the ratemap on the x axis
    biny: ndarray (nybins +1,)
        bin limits of the ratemap on the y axis
    """
    x_spikes, y_spikes = [], []

    # Find x, y position of each spike using spike times
    for s in spikes:
        array_pos = np.argmin(np.abs(time_array - s))
        x_spikes.append(x[array_pos])
        y_spikes.append(y[array_pos])
    x_spikes = np.array(x_spikes)
    y_spikes = np.array(y_spikes)
    h, binx, biny = np.histogram2d(x_spikes, y_spikes, bins=(x_size, y_size))
    # Gaussian filter
    if filter_result:
        h = gaussian_filter(h, sigma=2)
    # Transpose to change row col coordinates to positions
    return h.T, binx, biny


class OnlineRateMap(object):
    def __init__(
        self,
        spikes,
        position,
        size=(100, 100),
        x_range=(-100, 100),
        y_range=(-100, 100),
    ):
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
        self.x_pos_bins = np.linspace(x_range[0], x_range[1], num=size[0] + 1)
        self.y_pos_bins = np.linspace(y_range[0], y_range[1], num=size[1] + 1)
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
        pos = self.position[
            np.logical_and(self.position[:, 0] >= t_init, self.position[:, 0] < t_end),
            :,
        ]

        # Interpolate position for smooth plot
        if interp_factor != 1:
            t = pos[:, 0]
            x = pos[:, 1]
            y = pos[:, 2]
            t_inter = np.linspace(np.amin(t), np.amax(t), num=len(t) * interp_factor, endpoint=False)
            fx = interp1d(t, x)
            fy = interp1d(t, y)
            x_inter = fx(t_inter)
            y_inter = fy(t_inter)
            pos = np.stack([t_inter, x_inter, y_inter], axis=1)

        spk = self.spikes[np.logical_and(self.spikes >= t_init, self.spikes < t_end)]

        h_spk, bin_spk = np.histogram(spk, bins=np.linspace(t_init, t_end, num=pos.shape[0] + 1))

        # Convert position to indexes in the ratemap
        x_index = (pos[:, 1] - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * self.size[1]
        y_index = (pos[:, 2] - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * self.size[0]
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
        pos = self.position[
            np.logical_and(self.position[:, 0] >= t_init, self.position[:, 0] < t_end),
            :,
        ]

        # Interpolate position for smooth plot
        if interp_factor != 1:
            t = pos[:, 0]
            x = pos[:, 1]
            y = pos[:, 2]
            t_inter = np.linspace(np.amin(t), np.amax(t), num=len(t) * interp_factor, endpoint=False)
            fx = interp1d(t, x)
            fy = interp1d(t, y)
            x_inter = fx(t_inter)
            y_inter = fy(t_inter)
            pos = np.stack([t_inter, x_inter, y_inter], axis=1)

        spk = self.spikes[np.logical_and(self.spikes >= t_init, self.spikes < t_end)]

        h_spk, bin_spk = np.histogram(spk, bins=np.linspace(t_init, t_end, num=pos.shape[0] + 1))

        # Convert position to indexes in the ratemap
        x_index = (pos[:, 1] - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * self.size[1]
        y_index = (pos[:, 2] - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * self.size[0]
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
        self.nan = np.nan
        filtered_ratemap[nan_indexes] = self.nan
        return filtered_ratemap


def check_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def get_date_time():
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S-%f")[:-3]
    now = str(dt_string)
    return now


def dict_to_json(dict, path):
    with open(path, "w") as outfile:
        json.dump(dict, outfile)


def load_json(path):
    with open(path, "r") as infile:
        dict = json.load(infile)
    return dict
