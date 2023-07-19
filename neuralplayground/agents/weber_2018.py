"""
Implementation for Weber and Sprekeler 2018
Learning place cells, grid cells and invariances with excitatory and inhibitory plasticity
https://doi.org/10.7554/eLife.34560.001

This implementation can interact with environments from the package as shown in the examples jupyter notebook.
Check examples/testing_weber_model.ipynb
"""


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

from neuralplayground.plotting.plot_utils import make_plot_rate_map

from .agent_core import AgentCore


class Weber2018(AgentCore):
    """
    Implementation for Weber and Sprekeler 2018
    Learning place cells, grid cells and invariances with excitatory and inhibitory plasticity
    https://doi.org/10.7554/eLife.34560.001

    This implementation can interact with environments from the package as shown in the examples jupyter notebook.
    ...

    Attributes
    ----------
    mod_kwargs : dict
        Model parameters, please refer to ExcInhPlasticity.__init__ parameters
    inh_cell_list: ndarray
        Ni x (self.resolution**2) with tuning curves for each inhibitory neuron and each pixel in the 2D array
    exc_cell_list: ndarray
        Ne x (self.resolution**2) with tuning curves for each excitatory neuron and each pixel in the 2D array
    wi: ndarray
        (Ni, ) with inhibitory weights
    we: ndarray
        (Ne, ) with excitatory weights
    init_we_sum: float
        init normalization factor to maintain norm of excitatory weights constant
    obs_history: list
        List of past observations while interacting with the environment in the act method
    global_steps: int
        Record of number of updates done on the weights
    inh_rates_functions: list
        List of parameters of Gaussian functions in tuning curves for each inhibitory neuron
    exc_rates_functions: list
        List of parameters of Gaussian functions in tuning curves for each excitatory neuron

    Methods
    -------
    reset(self)
        Initialize weights, tuning curves and global counter for weights updates done
    generate_tuning_curves(self, n_curves, cov_scale, Nf, alpha)
        Generate tuning curves as a combination of Gaussians
        for a given number of neurons, covariance, number of Gaussian per tuning curve and height
    get_output_rates(self, pos)
        Compute firing rate of the output neuron for the specific position, equation 1
    get_rates(self, cell_list, pos)
        Get firing rate if input neurons at the given position by finding the pixel that is closest
        to the position given
    get_full_output_rate(self)
        It computes equation 1 (or 4) for all available pixels at the same time taking
        advantage of broadcasting
    update(self)
        Update weights using hebbian plasticity according to equation 2 for excitatory weights
        and equation 3 for inhibitory weights for a given position
    full_average_update(self, exc_normalization=True)
        Update weights using hebbian plasticity according to equation 2 for excitatory weights
        and equation 3 for inhibitory weights for all available positions in the grid
    full_update(self, exc_normalization=True)
        Update weights using hebbian plasticity according to equation 2 for excitatory weights
        and equation 3, but using self.update function as a sub routine
    plot_rates(self, save_path=None)
        Plot current rates and an example of inhibitory and excitatory neuron
    """

    def __init__(self, model_name: str = "ExcitInhibitoryplastic", **mod_kwargs):
        """
        Refer to Table 1 and Table 2 from the paper for best parameter selection

        Parameters
        ----------
        model_name : str
            Name of the specific instantiation of the ExcInhPlasticity class
        mod_kwargs : dict
            Dictionary with parameters of the model from Weber and Sprekeler 2018
            https://doi.org/10.7554/eLife.34560.001
            agent_step_size: float
                Size of movement by the agent within the environment
            exc_eta: float
                Excitatory learning rate (equation 2)
            inh_eta: float
                Inhibitory learning rate (equation 3)
            Ne: int
                Number of excitatory neurons (equation 1)
            Ni: int
                Number of inhibitory neurons (equation 1)
            Nef: int
                Number of Gaussians in the spatial tuning of excitatory neurons (Table 1)
            Nif: int
                Number of Gaussians in the spatial tuning of inhibitory neurons (Table 2)
            alpha_e: float
                Gaussian height of Gaussian functions in spatial tuning for excitatory neurons (equation 12)
            alpha_i: float
                Gaussian height of Gaussian functions in spatial tuning for excitatory neurons (equation 12)
            we_init: float
                Initial weights for excitatory neurons (Tables 1 and 2)
            wi_init: float
                Initial weights for inhibitory neurons (Tables 1 and 2)
            sigma_exc: float
                Standard deviation of Gaussian functions in excitatory tuning curves (equation 12)
            sigma_inh: float
                Standard deviation of Gaussian function in inhibitory tuning curves (equation 12)
            ro: float
                target rate for inhibitory neurons (equation 3)
            room_width: float
                room width specified by the environment (see examples/testing_weber_model.ipynb)
            room_depth: float
                room depth specified by the environment (see examples/testing_weber_model.ipynb)

        """
        super().__init__(model_name, **mod_kwargs)
        self.agent_step_size = mod_kwargs["agent_step_size"]
        self.metadata = {"mod_kwargs": mod_kwargs}
        self.etaexc = mod_kwargs["exc_eta"]
        self.etainh = mod_kwargs["inh_eta"]
        self.Ne = mod_kwargs["Ne"]
        self.Ni = mod_kwargs["Ni"]
        self.Nef = mod_kwargs["Nef"]
        self.Nif = mod_kwargs["Nif"]
        self.alpha_i = mod_kwargs["alpha_i"]
        self.alpha_e = mod_kwargs["alpha_e"]
        self.we_init = mod_kwargs["we_init"]
        self.wi_init = mod_kwargs["wi_init"]

        self.sigma_exc = mod_kwargs["sigma_exc"]
        self.sigma_inh = mod_kwargs["sigma_inh"]
        if "resolution" in mod_kwargs.keys():
            self.resolution = mod_kwargs["resolution"]
        else:
            self.resolution = 50
        if "disable_tqdm" in mod_kwargs.keys():
            self.disable_tqdm = mod_kwargs["disable_tqdm"]
        else:
            self.disable_tqdm = False

        self.room_width, self.room_depth = (
            mod_kwargs["room_width"],
            mod_kwargs["room_depth"],
        )
        self.ro = mod_kwargs["ro"]
        self.obs_history = []  # Initialize observation history to update weights later
        self.grad_history = []

        self.resolution = 100  # Number of pixels in the grid for the tuning functions

        self.resolution_width = self.resolution
        self.resolution_depth = int(self.resolution * (self.room_depth / self.room_width))

        self.x_array = np.linspace(-self.room_width / 2, self.room_width / 2, num=self.resolution_width)
        self.y_array = np.linspace(self.room_depth / 2, -self.room_depth / 2, num=self.resolution_depth)
        self.mesh = np.array(np.meshgrid(self.x_array, self.y_array))
        self.xy_combinations = self.mesh.T.reshape(-1, 2)
        self.reset()

    def reset(self):
        """
        Initialize weights, tuning curves and global counter for weights updates done
        """
        self.global_steps = 0  # Reset global steps
        self.obs_history = []  # Reset observation history
        self.grad_history = []
        # Initialize weights
        self.wi = np.random.uniform(
            low=self.wi_init - 0.05 * self.wi_init,
            high=self.wi_init + 0.05 * self.wi_init,
            size=(self.Ni,),
        )
        self.we = np.random.uniform(
            low=self.we_init - 0.05 * self.we_init,
            high=self.we_init + 0.05 * self.we_init,
            size=(self.Ne,),
        )

        # Initialize tuning functions
        self.inh_rates_functions, self.inh_cell_list = self.generate_tuning_curves(
            n_curves=self.Ni, cov_scale=self.sigma_inh, Nf=self.Nif, alpha=self.alpha_i
        )
        self.exc_rates_functions, self.exc_cell_list = self.generate_tuning_curves(
            n_curves=self.Ne, cov_scale=self.sigma_exc, Nf=self.Nef, alpha=self.alpha_e
        )

        self.init_we_sum = np.sqrt(np.sum(self.we**2))  # Keep track of normalization constant

    def generate_tuning_curves(self, n_curves: int, cov_scale: float, Nf: int, alpha: float):
        """
        Generate tuning curves as a combination of Gaussians
        for a given number of neurons, covariance, number of Gaussian per tuning curve and height

        Parameters
        ----------
        n_curves: int
            Number of neurons (Ne or Ni in __init__)
        cov_scale: float
            Standard deviation of Gaussian functions of each neuron tuning curves (exc_sigma or inh_sigma in __init__)
        Nf: int
            Number of Gaussians in the spatial tuning of each neuron (Nfe or Nfi in __init__)
        alpha: float
            Height of Gaussian functions in spatial tuning for neurons (alpha_i or alpha_e in __init__)

        Returns
        -------
        function_list: list
            List with parameters of each gaussian in the tuning curve
        cell_list: numpy ndarray
            n_curves x (self.resolution**2) with tuning curves for each neuron and each pixel in the 2D array
        """
        width_limit = self.room_width / 2.0
        depth_limit = self.room_depth / 2.0
        cell_list = []  # This will become a numpy array with the tuning curves in 2D (images)
        function_list = []  # List with parameters of each gaussian in the tuning curve
        for i in tqdm(range(n_curves), disable=self.disable_tqdm):
            gauss_list = []
            cell_i = 0
            for j in range(Nf):
                # The +0.2 in the following is to avoid border effects
                mean1 = np.random.uniform(-width_limit * (1 + 0.2), width_limit * (1 + 0.2))  # Sample means
                mean2 = np.random.uniform(-depth_limit * (1 + 0.2), depth_limit * (1 + 0.2))
                room_scale = np.max(np.array([self.room_width, self.room_width]))
                cov = np.diag(np.multiply(cov_scale, np.array([room_scale, room_scale])) ** 2)
                mean = np.array([mean1, mean2])
                rv = multivariate_normal(mean, cov)  # Generate gaussian
                gauss_list.append([mean, cov])
                normalization_constant = 2 * np.pi * np.sqrt(np.linalg.det(cov))  # normalization constant (eq 13)
                cell_i += rv.pdf(self.xy_combinations) * normalization_constant * alpha  # See equation 13
            function_list.append(gauss_list)
            cell_list.append(cell_i)
        cell_list = np.array(cell_list)
        return function_list, cell_list

    def get_output_rates(self, pos: np.ndarray):
        """
        Compute firing rate of the output neuron for the specific position, equation 1

        Parameters
        ----------
        pos : ndarray
            (2,) array with x, y coordinates

        Returns
        -------
        r_out : float
            output neuron firing rate in equation 1 (or equation 4)
        """
        exc_rates = self.get_rates(self.exc_cell_list, pos)
        inh_rates = self.get_rates(self.inh_cell_list, pos)

        r_out = self.we.T @ exc_rates - self.wi.T @ inh_rates
        r_out = np.clip(r_out, a_min=0, a_max=np.amax(r_out))
        return r_out

    def get_rates(self, cell_list: np.ndarray, pos: np.ndarray):
        """
        Get firing rate if input neurons at the given position by finding the pixel that is closest
        to the position given

        Parameters
        ----------
        cell_list : numpy ndarray
            n_curves x (self.resolution**2) with tuning curves for each neuron and each pixel in the 2D array
        pos : ndarray
            (2,) array with x, y coordinates

        Returns
        -------
        rout : ndarray
            (number_of_neurons, ) array with the output firing rate for each of the tuning curves at position pos
        """
        diff = self.xy_combinations - pos[np.newaxis, ...]
        dist = np.sum(diff**2, axis=1)
        index = np.argmin(dist)  # Get the pixel closest to the pos
        input_rates = []
        for i in range(cell_list.shape[0]):
            input_rates.append(cell_list[i, index])  # Firing rate for each cell
        input_rates = np.array(input_rates)
        input_rates = np.clip(input_rates, a_min=0, a_max=np.amax(input_rates))  # negatives to zero
        return input_rates

    def get_full_output_rate(self):
        """
        It computes equation 1 (or 4) for all available pixels at the same time taking
        advantage of broadcasting

        Returns
        -------
        r_out: ndarray
            (self.resolution**2, ) with output firing rate at each available position
        """
        r_out = self.we.T @ self.exc_cell_list - self.wi.T @ self.inh_cell_list
        r_out = np.clip(r_out, a_min=0, a_max=np.amax(r_out))
        return r_out

    def update(self, exc_normalization: bool = True, pos: np.ndarray = None):
        """
        Update weights using hebbian plasticity according to equation 2 for excitatory weights
        and equation 3 for inhibitory weights for a given position

        Parameters
        ----------
        exc_normalization : bool
            True if excitatory weights are normalized after each update
        pos : float
            Run update for a given position. Defaults to None, then it is replaced with the
            last position in obs_history
        """
        if pos is None:
            pos = self.obs_history[-1]
        r_out = self.get_output_rates(pos)

        # Excitatory weights update (eq 2)
        delta_we = self.etaexc * self.get_rates(self.exc_cell_list, pos=pos) * r_out
        # Inhibitory weights update (eq 3)
        delta_wi = self.etainh * self.get_rates(self.inh_cell_list, pos=pos) * (r_out - self.ro)
        self.grad_history.append(np.sqrt(np.sum(delta_we**2) + np.sum(delta_wi**2)))

        self.we = self.we + delta_we
        self.wi = self.wi + delta_wi

        if exc_normalization:
            self.we = self.init_we_sum / np.sqrt(np.sum(self.we**2)) * self.we

        self.we = np.clip(self.we, a_min=0, a_max=np.amax(self.we))  # Negative weights to zero
        self.wi = np.clip(self.wi, a_min=0, a_max=np.amax(self.wi))
        return {"delta_we": delta_we, "delta_wi": delta_wi}

    def full_average_update(self, exc_normalization: bool = True):
        """
        Update weights using hebbian plasticity according to equation 2 for excitatory weights
        and equation 3 for inhibitory weights for all available positions in the grid

        Parameters
        ----------
        exc_normalization : bool
            True if excitatory weights are normalized after each update
        """
        r_out = self.get_full_output_rate()
        r_out = r_out[..., np.newaxis]

        # Excitatory weights update (eq 2)
        delta_we = self.etaexc * (self.exc_cell_list @ r_out) / self.resolution**2
        # Inhibitory weights update (eq 3)
        delta_wi = self.etainh * (self.inh_cell_list @ (r_out - self.ro)) / self.resolution**2

        self.we = self.we + delta_we[:, 0]
        self.wi = self.wi + delta_wi[:, 0]

        if exc_normalization:
            self.we = self.init_we_sum / np.sqrt(np.sum(self.we**2)) * self.we

        self.we = np.clip(self.we, a_min=0, a_max=np.amax(self.we))  # Negative weights to zero
        self.wi = np.clip(self.wi, a_min=0, a_max=np.amax(self.wi))

    def full_update(self, exc_normalization: bool = True):
        """
        Update weights using hebbian plasticity according to equation 2 for excitatory weights
        and equation 3, but using self.update function as a sub routine

        Parameters
        ----------
        exc_normalization : bool
            True if excitatory weights are normalized after each update
        """
        random_permutation = np.arange(self.xy_combinations.shape[0])
        xy_array = self.xy_combinations[random_permutation, :]  # All points (x, y) in the grid
        for i in range(self.xy_combinations.shape[0]):
            self.update(exc_normalization=exc_normalization, pos=xy_array[i, :])

    def get_rate_map_matrix(
        self,
    ):
        """
        Get the ratemap matrix of the network

        Returns
        -------
        ratemap_matrix : ndarray
            (self.resolution_width, self.resolution_depth) with the ratemap matrix
        """
        r_out_im = self.get_full_output_rate()
        r_out_im = r_out_im.reshape((self.resolution_width, self.resolution_depth))
        return r_out_im

    def plot_rate_map(self, save_path: str = None, ax: mpl.axes.Axes = None):
        """
        Plot current rates and an example of inhibitory and excitatory neuron

        Parameters
        ----------
        save_path : str
            Path to save the figure. Default None, it doesn't save the figure
        ax : ndarray of matplotlib.axis
            (3,) with 3 axis to make plots from matplotlib, if None it will create an entire figure
        """
        if ax is None:
            f, ax = plt.subplots()
        r_out_im = self.get_full_output_rate()
        r_out_im = r_out_im.reshape((self.resolution_width, self.resolution_depth))
        make_plot_rate_map(r_out_im.T, ax, "Out rate", "width", "depth", "Firing rate")

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close("all")
        else:
            return ax

    def plot_all_rates(self, save_path: str = None, ax: mpl.axes.Axes = None):
        """
        Plot current rates and an example of inhibitory and excitatory neuron

        Parameters
        ----------
        save_path : str
            Path to save the figure. Default None, it doesn't save the figure
        ax : ndarray of matplotlib.axis
            (3,) with 3 axis to make plots from matplotlib, if None it will create an entire figure
        """
        if ax is None:
            f, ax = plt.subplots(1, 3, figsize=(14, 5))

        r_out_im = self.get_full_output_rate()
        r_out_im = r_out_im.reshape((self.resolution_width, self.resolution_depth))
        exc_im = self.exc_cell_list[np.random.choice(np.arange(self.exc_cell_list.shape[0])), ...].reshape(
            (self.resolution_width, self.resolution_depth)
        )
        inh_im = self.inh_cell_list[np.random.choice(np.arange(self.inh_cell_list.shape[0])), ...].reshape(
            (self.resolution_width, self.resolution_depth)
        )
        make_plot_rate_map(exc_im.T, ax[0], "Exc rates", "width", "depth", "Firing rate")
        make_plot_rate_map(inh_im.T, ax[1], "Inh rates", "width", "depth", "Firing rate")
        make_plot_rate_map(r_out_im.T, ax[2], "Out rate", "width", "depth", "Firing rate")

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close("all")
        else:
            return ax
