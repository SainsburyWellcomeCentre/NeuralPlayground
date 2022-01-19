"""
Implementation for Weber and Sprekeler 2018
Learning place cells, grid cells and invariances with excitatory and inhibitory plasticity
https://doi.org/10.7554/eLife.34560.001

This implementation can interact with environments from the package as shown in the examples jupyter notebook.
Check examples/testing_weber_model.ipynb
"""

import sys
sys.path.append("../")
import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

from models.core import NeuralResponseModel
from environments.environments.simple2d import Simple2D, Sargolini2006, BasicSargolini2006


class ExcInhPlasticity(NeuralResponseModel):
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
        List of parameters of Gaussian functions in tuning curves for inhibitory neurons
    exc_rates_functions: list
        List of parameters of Gaussian functions in tuning curves for excitatory neurons

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

    def __init__(self, model_name="ExcitInhibitoryplastic", **mod_kwargs):
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

        self.room_width, self.room_depth = mod_kwargs["room_width"], mod_kwargs["room_depth"]
        self.ro = mod_kwargs["ro"]
        self.obs_history = []  # Initialize observation history to update weights later

        self.resolution = 50  # Number of pixels in the grid for the tuning functions
        self.x_array = np.linspace(-self.room_width/2, self.room_width/2, num=self.resolution)
        self.y_array = np.linspace(self.room_depth/2, -self.room_depth/2, num=self.resolution)
        self.mesh = np.array(np.meshgrid(self.x_array, self.y_array))
        self.xy_combinations = self.mesh.T.reshape(-1, 2)
        self.reset()

    def reset(self):
        """
        Initialize weights, tuning curves and global counter for weights updates done
        """
        self.global_steps = 0  # Reset global steps
        self.obs_history = []  # Reset observation history
        # Initialize weights
        self.wi = np.random.uniform(low=self.wi_init-0.05*self.wi_init,
                                    high=self.wi_init+0.05*self.wi_init, size=(self.Ni,))
        self.we = np.random.uniform(low=self.we_init-0.05*self.we_init,
                                    high=self.we_init+0.05*self.we_init, size=(self.Ne,))

        # Initialize tuning functions
        self.inh_rates_functions, self.inh_cell_list = self.generate_tuning_curves(n_curves=self.Ni,
                                                                                   cov_scale=self.sigma_inh,
                                                                                   Nf=self.Nif,
                                                                                   alpha=self.alpha_i)
        self.exc_rates_functions, self.exc_cell_list = self.generate_tuning_curves(n_curves=self.Ne,
                                                                                   cov_scale=self.sigma_exc,
                                                                                   Nf=self.Nef,
                                                                                   alpha=self.alpha_e)

        self.init_we_sum = np.sqrt(np.sum(self.we**2))  # Keep track of normalization constant

    def generate_tuning_curves(self, n_curves, cov_scale, Nf, alpha):
        """
        Generate tuning curves as a combination of Gaussians
        for a given number of neurons, covariance, number of Gaussian per tuning curve and height

        Parameters
        ----------
        n_curves: int
            Number of neurons (Ne or Ni in __init__)
        cov_scale: float
            Standard deviation of Gaussian functions in neurons tuning curves (exc_sigma or inh_sigma in __init__)
        Nf: int
            Number of Gaussians in the spatial tuning of neurons (Nfe or Nfi in __init__)
        alpha: float
            Gaussian height of Gaussian functions in spatial tuning for neurons (alpha_i or alpha_e in __init__)

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
        for i in tqdm(range(n_curves)):
            gauss_list = []
            cell_i = 0
            for j in range(Nf):
                # The +0.2 in the following is to avoid border effects
                mean1 = np.random.uniform(-width_limit*(1+0.2), width_limit*(1+0.2))  # Sample means
                mean2 = np.random.uniform(-depth_limit*(1+0.2), depth_limit*(1+0.2))
                cov = np.diag(np.multiply(cov_scale, np.array([self.room_width, self.room_depth]))**2)
                mean = np.array([mean1, mean2])
                rv = multivariate_normal(mean, cov)  # Generate gaussian
                gauss_list.append([mean, cov])
                normalization_constant = 2*np.pi*np.sqrt(np.linalg.det(cov))  # normalization constant (eq 13)
                cell_i += rv.pdf(self.xy_combinations)*normalization_constant*alpha  # See equation 13
            function_list.append(gauss_list)
            cell_list.append(cell_i)
        cell_list = np.array(cell_list)
        return function_list, cell_list

    def get_output_rates(self, pos):
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

    def get_rates(self, cell_list, pos):
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
        rout = []
        for i in range(cell_list.shape[0]):
            rout.append(cell_list[i, index])  # Firing rate for each cell
        rout = np.array(rout)
        rout = np.clip(rout, a_min=0, a_max=np.amax(rout))  # negatives to zero
        return rout

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

    def update(self, exc_normalization=True, pos=None):
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
        delta_we = self.etaexc*self.get_rates(self.exc_cell_list, pos=pos)*r_out
        # Inhibitory weights update (eq 3)
        delta_wi = self.etainh*self.get_rates(self.inh_cell_list, pos=pos)*(r_out - self.ro)

        self.we = self.we + delta_we
        self.wi = self.wi + delta_wi

        if exc_normalization:
            self.we = self.init_we_sum/np.sqrt(np.sum(self.we**2))*self.we

        self.we = np.clip(self.we, a_min=0, a_max=np.amax(self.we))  # Negative weights to zero
        self.wi = np.clip(self.wi, a_min=0, a_max=np.amax(self.wi))

    def full_average_update(self, exc_normalization=True):
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
        delta_we = self.etaexc*(self.exc_cell_list @ r_out)/self.resolution**2
        # Inhibitory weights update (eq 3)
        delta_wi = self.etainh*(self.inh_cell_list @ (r_out-self.ro))/self.resolution**2

        self.we = self.we + delta_we[:, 0]
        self.wi = self.wi + delta_wi[:, 0]

        if exc_normalization:
            self.we = self.init_we_sum/np.sqrt(np.sum(self.we**2))*self.we

        self.we = np.clip(self.we, a_min=0, a_max=np.amax(self.we))  # Negative weights to zero
        self.wi = np.clip(self.wi, a_min=0, a_max=np.amax(self.wi))

    def full_update(self, exc_normalization=True):
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

    def plot_rates(self, save_path=None):
        """
        Plot current rates and an example of inhibitory and excitatory neuron

        Parameters
        ----------
        save_path: str
            Path to save the figure. Default None, it doesn't save the figure
        """
        f, ax = plt.subplots(1, 3, figsize=(14, 5))

        r_out_im = self.get_full_output_rate()
        r_out_im = r_out_im.reshape((self.resolution, self.resolution))

        exc_im = self.exc_cell_list[np.random.choice(np.arange(self.exc_cell_list.shape[0]))
            , ...].reshape((self.resolution, self.resolution))
        inh_im = self.inh_cell_list[np.random.choice(np.arange(self.exc_cell_list.shape[0]))
            , ...].reshape((self.resolution, self.resolution))

        ax[0].imshow(exc_im, cmap="Reds")
        ax[0].set_title("Exc rates", fontsize=14)
        ax[1].imshow(inh_im, cmap="Blues")
        ax[1].set_title("Inh rates", fontsize=14)
        im = ax[2].imshow(r_out_im)
        ax[2].set_title("Out rate", fontsize=14)
        cbar = plt.colorbar(im, ax=ax[2])

        if not save_path is None:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close("all")
        else:
            plt.show()


if __name__ == "__main__":

    """ This part of the code is just for testing and it could be broke
    For examples on how to use this class please refer to the examples in examples/testing_weber_model.ipynb 
     """
    run_raw_data = False

    if run_raw_data:
        data_path = "/home/rodrigo/HDisk/8F6BE356-3277-475C-87B1-C7A977632DA7_1/all_data/"

        session = {"rat_id": "11016", "sess": "31010502"}

        env = Sargolini2006(data_path=data_path,
                            verbose=False,
                            session=session,
                            time_step_size=None,
                            agent_step_size=None)

        exc_eta = 6.7e-5
        inh_eta = 2.7e-4
        model_name = "model_example"
        sigma_exc = 0.05
        sigma_inh = 0.1
        Ne = 4900
        Ni = 1225
        Nef = 1
        Nif = 1
        agent_step_size = 0.1
        alpha_i = 1
        alpha_e = 1

        print("init cells")
        agent = ExcInhPlasticity(model_name=model_name, exc_eta=exc_eta, inh_eta=inh_eta, sigma_exc=sigma_exc,
                                 sigma_inh=sigma_inh, Ne=Ne, Ni=Ni, agent_step_size=agent_step_size, ro=1,
                                 Nef=Nef, Nif=Nif, room_width=env.room_width, room_depth=env.room_depth,
                                 alpha_i=alpha_i, alpha_e=alpha_e)

        print("Plotting rate")
        agent.plot_rates("figures/init_rates.pdf")

        print("running updates")
        n_steps = 30000
        # Initialize environment

        total_iters = 0

        all_sessions = {"11016": ['02020502', '25010501', '28010501', '29010503', '31010502'],  # 5/6
                        "10884": ['01080402', '02080405', '03080402', '03080405', '03080409', '04080402', '05080401',
                                   '08070402', '08070405', '09080404', '13070402', '14070405', '16070401', '19070401',
                                   '21070405', '24070401', '31070404'],  # 22/6
                        "10704": ['06070402', '07070402', '07070407', '08070402', '19070402', '20060402',
                                  '20070402', '23060402', '25060402', '26060402'],  # 32/6
                        "11084": ['01030503', '02030502', '03020501', '08030506', '09030501', '09030503', '10030502',
                                  '23020502', '24020502', '28020501'],  # 42/6
                        "11265": ['01020602', '02020601', '03020601', '06020601', '07020602', '09020601', '13020601',
                                  '16030601', '16030604', '31010601'],  # 52/6
                        "11207": ['03060501', '04070501', '05070501', '06070501', '07070501', '08060501', '08070504',
                                  '09060501']}  # 60/6

        for rat_id, session_list in all_sessions.items():
            for j, sess in enumerate(session_list):
                session = {"rat_id": rat_id, "sess": sess}
                obs, state = env.reset(sess=session)
                print("Running sess", session)
                for i in tqdm(range(n_steps)):
                    # Observe to choose an action
                    obs = obs[:2]
                    action = agent.act(obs)
                    rate = agent.update()
                    # Run environment for given action
                    obs, state, reward = env.step(action)
                    total_iters += 1
                agent.plot_rates(save_path="figures/iter_"+str(total_iters)+".pdf")

        print("plotting results")
        agent.plot_rates()

    else:
        data_path = "../environments/experiments/Sargolini2006/"
        env = BasicSargolini2006(data_path=data_path,
                                 time_step_size=0.1,
                                 agent_step_size=None)
        exc_eta = 2e-4
        inh_eta = 8e-4
        model_name = "model_example"
        sigma_exc = np.array([0.05, 0.05])
        sigma_inh = np.array([0.1, 0.1])
        Ne = 4900
        Ni = 1225
        Nef = 1
        Nif = 1
        alpha_i = 1
        alpha_e = 1
        we_init = 1.0
        wi_init = 1.5

        agent_step_size = 0.1

        agent = ExcInhPlasticity(model_name=model_name, exc_eta=exc_eta, inh_eta=inh_eta, sigma_exc=sigma_exc,
                                 sigma_inh=sigma_inh, Ne=Ne, Ni=Ni, agent_step_size=agent_step_size, ro=1,
                                 Nef=Nef, Nif=Nif, room_width=env.room_width, room_depth=env.room_depth,
                                 alpha_i=alpha_i, alpha_e=alpha_e, we_init=we_init, wi_init=wi_init)

        agent.plot_rates()

        print("debug")

        plot_every = 10
        total_iters = 0

        obs, state = env.reset()
        #for i in tqdm(range(env.total_number_of_steps)):
        for i in tqdm(range(5000)):
            # Observe to choose an action
            obs = obs[:2]
            action = agent.act(obs)
            # rate = agent.update()
            agent.full_update()
            # Run environment for given action
            obs, state, reward = env.step(action)
            total_iters += 1
            if i % plot_every == 0:
                agent.plot_rates(save_path="figures/pre_processed_iter_"+str(i)+".pdf")
