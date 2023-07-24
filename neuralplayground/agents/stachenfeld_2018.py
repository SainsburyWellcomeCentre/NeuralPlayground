"""
Implementation for 2017 by Kimberly L. Stachenfeld1,2,*, Matthew M. Botvinick1,3, and Samuel J. Gershman
The hippocampus as a predictive map
https://doi.org/10.1101/097170;

This implementation can interact with environments from the package as shown in the examples jupyter notebook.
Check examples/Stachenfeld_2018_example.ipynb
"""

import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .agent_core import AgentCore

sys.path.append("../")


class Stachenfeld2018(AgentCore):
    """
    Implementation for SR 2017 by Kimberly L. Stachenfeld1,2,*, Matthew M. Botvinick1,3, and Samuel J. Gershman4
    The hippocampus as a predictive map
    https://doi.org/10.1101/097170;

    This implementation can interact with environments from the package as shown in the examples jupyter notebook.
    Check examples/SRexample.ipynb
    ----
    Attributes
    ---------
     mod_kwargs : dict
        Model parameters
        gamma: scalar,
            discounting factor
        learning_rate: scalar,
            scaling the magnitude of the TD update
        t_episode: scalar,
            maximum number of timesteps in one episode
        threoshold: scalar,
            upper bound for the update size
        n_episode: scalar,
            number of episodes
        transmat: (n_state, n_state)
            numpy array, transition matrix
        room_width: float
                    room width specified by the environment (see examples/examples/SRexample.ipynb)
        room_depth: float
                    room depth specified by the environment (see examples/examples/SRexample.ipynb)
        state_density: float
            density of SR-agent states (should be proportional to the step-size)
        twoD: bool
                When true creates a (n_state, n_state) transition array for a rectangular 2D state space.


    Methods
    ---------
    reset(self):
        Initialize the successor matrices, normalized transition matrix and observation variables ( history and initialisation)
    obs_to_state(self, pos):
        Converts the agent's position in the environment to the agent's position in the SR-agent state space.
    act: float
        The base model executes one of four action (up-down-right-left) with equal probability.
        This is used to move on the rectangular environment states space (transmat).
    get_T_from_M(self, M):
        Compute the transition matrix from the computationally simulated successor representation matrix M
    create_transmat(self, state_density, name_env, plotting_varible=True):
        Creates the normalised transition matrix for a rectangular environment '2D_env'
    update_successor_rep(self):
        Compute the successor representation matrix using geometric sums
    successor_rep_sum(self):
        Compute the successor representation using successive additive update
    update_successor_rep_td_full(self):
        Compute the successor representation matrix using TD learning
    update(self):
        Compute the successor representation matrix using TD learning while interacting with the environement
    plot_transition(self, matrix, save_path=None, ax=None):
        Plot the input matrix and compare it to the transition matrix from the
        rectangular environment states space (rectangular- transmat).
    plot_eigen(self,matrix, save_path, ax=None):
        Plot the matrix and the 4 largest modes of its eigen-decomposition
    """

    def __init__(self, model_name: str = "SR", **mod_kwargs):
        """
        Parameters
        ----------
        model_name : str
            Name of the specific instantiation of the ExcInhPlasticity class
        mod_kwargs : dict
            gamma: scalar,
                 discounting factor
            learning_rate: scalar,
                scaling the magnitude of the TD update
            t_episode: scalar,
                maximum number of timesteps in one episode
            threshold: scalar,
               upper bound for the update size
            n_episode: scalar,
                number of episodes
            twoD: bool
                When true creates a (n_state, n_state) transition array for a rectangular 2D state space.
            room_width: float
                room width specified by the environment (see examples/examples/SRexample.ipynb)
            room_depth: float
                room depth specified by the environment (see examples/examples/SRexample.ipynb)
            state_density: float
                density of SR-agent states (should be proportional to the step-size)
        """
        super().__init__(model_name, **mod_kwargs)
        self.metadata = {"mod_kwargs": mod_kwargs}
        self.obs_history = []  # Initialize observation history to update weights later
        self.grad_history = []
        self.gamma = mod_kwargs["discount"]
        self.threshold = mod_kwargs["threshold"]
        self.learning_rate = mod_kwargs["lr_td"]
        self.t_episode = mod_kwargs["t_episode"]
        self.n_episode = mod_kwargs["n_episode"]
        self.room_width = mod_kwargs["room_width"]
        self.room_depth = mod_kwargs["room_depth"]
        self.state_density = mod_kwargs["state_density"]
        twoD = mod_kwargs["twoD"]
        self.inital_obs_variable = None

        self.reset()
        # Variables for the SR-agent state space
        self.resolution_d = int(self.state_density * self.room_depth)
        self.resolution_w = int(self.state_density * self.room_width)
        self.x_array = np.linspace(-self.room_width / 2, self.room_width / 2, num=self.resolution_d)
        self.y_array = np.linspace(self.room_depth / 2, -self.room_depth / 2, num=self.resolution_w)
        self.mesh = np.array(np.meshgrid(self.x_array, self.y_array))
        self.xy_combinations = self.mesh.T.reshape(-1, 2)
        self.w = int(self.room_width * self.state_density)
        self.l = int(self.room_depth * self.state_density)
        self.n_state = int(self.l * self.w)
        self.obs_history = []
        if twoD:
            self.create_transmat(self.state_density, "2D_env")

    def reset(self):
        """
        Initialize the successor matrices, normalized transition matrix and observation variables (history and initialisation)
        """

        self.srmat = []
        self.srmat_sum = []
        self.srmat_ground = []
        self.transmat_norm = []
        self.inital_obs_variable = None
        self.obs_history = []  # Reset observation history

    def obs_to_state(self, pos: np.ndarray):
        """
        Converts the agent's position in the environment to the agent's position in the SR-agent state space.

        Parameters
        ----------
        pos: array (2,1)
            array containing the observed position of the agent in the environment

        Returns
        -------
        curr_state: int
            integer corresponding to the position in the SR-agent state space


        """
        np.arange(self.n_state).reshape(self.l, self.w)

        diff = self.xy_combinations - pos[np.newaxis, ...]
        dist = np.sum(diff**2, axis=1)
        index = np.argmin(dist)
        curr_state = index
        return curr_state

    def act(self, obs):
        """
        The base model executes one of four action (up-down-right-left) with equal probability.
        This is used to move on the rectangular environment states space (transmat).
        Parameters
        ----------
        obs: array (2,1)
            Observation from the environment class needed to choose the right action (Here the position).
        Returns
        -------
        action : array (2,1)
            Action value (Direction of the agent step) in this case executes one of
            four action (up-down-right-left) with equal probability.
        """
        self.obs_history.append(obs)
        if len(self.obs_history) >= 1000:
            self.obs_history = [
                obs,
            ]
        arrow = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        action = np.random.normal(scale=0.1, size=(2,))
        diff = action - arrow
        dist = np.sum(diff**2, axis=1)
        index = np.argmin(dist)
        action = arrow[index]
        self.next_state = self.obs_to_state(obs)
        action = np.array(action)
        return action

    def get_T_from_M(self, M: np.ndarray):
        """
        Compute the transition matrix from the computationally simulated successor matrix M
        Parameters
        ----------
        M: array (n_state,n_state)
            Successor representation matrix M

        Returns
        -------
        T: array (n_state,n_state)
             The computed transition matrix from the successor representation matrix M
        """
        T = (1 / self.gamma) * np.linalg.inv(M) @ (M - np.eye(self.n_state))
        return T

    def create_transmat(self, state_density: float, name_env: str, plotting_variable: bool = False):
        """
        Creates the normalised transition matrix for a rectangular environment '2D_env'

        Parameters
        ----------
            state_density: float
                density of SR-agent states (should be proportional to the step-size)
            name_env: string
                name of the environment to create ( There is only one environment type for now)
                If a new state space type is added please update the action function accordingly
            plotting_variable: bool
                If True: plots the normalised transition matrix
        Returns:
        -------
            transmat_norm: array (n_state,n_state)
                Normalised transition matrix

        """

        if name_env == "2D_env":
            adjmat_triu = np.zeros((self.n_state, self.n_state))
            node_layout = np.arange(self.n_state).reshape(self.l, self.w)
            np.linspace(0, np.min([self.l / self.w, 1]), num=self.l)
            np.linspace(0, np.min([self.w / self.l, 1]), num=self.w)
            self.xy = []

            for i in range(self.l):
                for j in range(self.w):
                    s = node_layout[i, j]
                    neighbours = []
                    if i - 1 >= 0:
                        neighbours.append(node_layout[i - 1, j])
                    if i + 1 < self.l:
                        neighbours.append(node_layout[i + 1, j])
                    if j - 1 >= 0:
                        neighbours.append(node_layout[i, j - 1])
                    if j + 1 < self.w:
                        neighbours.append(node_layout[i, j + 1])
                    adjmat_triu[s, neighbours] = 1

            transmat = adjmat_triu + adjmat_triu.T
            transmat = np.array(transmat, dtype=np.float64)
            row_sums = np.sum(transmat, axis=1)
            row_sums[row_sums == 0] = 1
            self.transmat_norm = transmat / row_sums.reshape(-1, 1)

        # Initial srmat
        self.srmat = np.eye(self.n_state)

        if plotting_variable is True:
            f, ax = plt.subplots(1, 1, figsize=(14, 5))
            ax.imshow(self.transmat_norm, interpolation="nearest", cmap="jet")

        return self.transmat_norm

    def update_successor_rep(self):
        """
        Compute the successor representation matrix using geometric sums.

        Returns:
        -------
            srmat_ground: (n_state, n_state) numpy array,
                Successor representation matrix
        """
        transmat_type = np.array(self.transmat_norm, dtype=np.float64)

        self.srmat_ground = np.linalg.inv(np.eye(self.n_state) - self.gamma * transmat_type)
        return self.srmat_ground

    def successor_rep_sum(self):
        """
        Compute the successor representation using successive additive update

        Returns:
        -------
            srmat_sum: (n_state, n_state) numpy array, successor representation matrix
        """

        self.srmat_sum = np.zeros_like(self.transmat_norm)
        keep_going = True
        while keep_going:
            new_srmat = self.gamma * self.transmat_norm.dot(self.srmat_sum) + np.eye(self.n_state)
            update = new_srmat - self.srmat_sum
            self.srmat_sum = new_srmat
            if np.max(np.abs(update)) < self.threshold:
                keep_going = False

        return self.srmat_sum

    def update(self):
        """
        Compute the successor representation matrix using TD learning while interacting with the environement

        Returns:
        -------
            srmat: (n_state, n_state) successor representation matrix
        """

        if self.inital_obs_variable is None:
            self.curr_state = self.next_state
            self.inital_obs_variable = True

        next_state = self.next_state
        self.n_state = self.transmat_norm.shape[0]
        a = np.array(self.curr_state)
        x = a.flatten()
        b = np.eye(self.n_state)[x, : self.n_state]
        L = b.reshape(a.shape + (self.n_state,))
        curr_state_vec = L

        update_val = curr_state_vec + self.gamma * self.srmat[:, next_state] - self.srmat[:, self.curr_state]
        self.srmat[:, self.curr_state] = self.srmat[:, self.curr_state] + self.learning_rate * update_val

        self.grad_history.append(np.sqrt(np.sum(update_val**2)))
        self.curr_state = next_state

        return self.srmat

    def update_successor_rep_td_full(self):
        """
        Compute the successor representation matrix using TD learning

        Returns:
        ----------
            srmat_full: (n_state, n_state)
                successor representation matrix

        """
        random_state = np.random.RandomState(1234)

        t_elapsed = 0
        srmat0 = np.eye(self.n_state)
        srmat_full = srmat0.copy()
        for i in range(self.n_episode):
            curr_state = random_state.randint(self.n_state)
            for j in range(self.t_episode):
                a = np.array([curr_state])
                x = a.flatten()
                b = np.eye(self.n_state)[x, : self.n_state]
                L = b.reshape(a.shape + (self.n_state,))

                curr_state_vec = L
                random_state.multinomial(1, self.transmat_norm[curr_state, :])
                next_state = np.where(random_state.multinomial(1, self.transmat_norm[curr_state, :]))[0][0]

                srmat_full[:, curr_state] = srmat_full[:, curr_state] + self.learning_rate * (
                    curr_state_vec + self.gamma * srmat_full[:, next_state] - srmat_full[:, curr_state]
                )
                curr_state = next_state
                t_elapsed += 1

        return srmat_full

    def plot_transition(self, matrix: np.ndarray, save_path: str = None, ax: mpl.axes.Axes = None):
        """
        Plot the input matrix and compare it to the transition matrix from the rectangular
        environment states space (rectangular- transmat).
        (If a new state space type is added please update this function)
        Parameters
        ----------
        matrix: array
            The matrix that will be plotted
        save_path: string
            Path to save the plot
        """
        evals, evecs = np.linalg.eig(matrix)
        if ax is None:
            f, ax = plt.subplots(1, 2, figsize=(14, 5))
            ax[0].imshow(self.transmat_norm, cmap="jet")
            ax[1].imshow(matrix, cmap="jet")
        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close("all")
        return ax

    def plot_eigen(self, matrix: np.ndarray, save_path: str, eigen=(0, 1), ax: mpl.axes.Axes = None):
        """ "
        Plot the matrix and the 4 largest modes of its eigen-decomposition

        Parameters
        ----------
        matrix: array
            The matrix that will be plotted
        eigen:  np.ndarray
            Which eigenvectors you would like to plot
        save_path: string
            Path to save the plot
        """

        evals, evecs = np.linalg.eig(matrix)

        if ax is None:
            f, ax = plt.subplots(1, len(eigen), figsize=(4 * len(eigen), 5))
            if len(eigen) == 1:
                evecs_0 = evecs[:, eigen[0]].reshape(self.l, self.w).real
                im = ax.imshow(evecs_0, cmap="jet")
                plt.colorbar(im, ax=ax)
            else:
                for i, eig in enumerate(eigen):
                    evecs_0 = evecs[:, eig].reshape(self.l, self.w).real
                    im = ax[i].imshow(evecs_0, cmap="jet")
                plt.colorbar(im, ax=ax[i])
        if save_path is None:
            pass
        else:
            plt.savefig(save_path, bbox_inches="tight")
        return ax
