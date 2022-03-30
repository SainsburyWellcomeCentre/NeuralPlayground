"""
Implementation for SR Kim
"""
import sys
sys.path.append("../")
import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

from .modelcore import NeuralResponseModel
from ..envs.arenas.simple2d import Simple2D


class SR(NeuralResponseModel):
    """
    Attributes

    gamma: scalar, discounting factor
    learning_rate: scalar, scaling the magnitude of the TD update
    random_state: random state object, e.g. np.random.RandomState(seed))
    set for reproducibility
    t_episode: scalar, maximum number of timesteps in one episode
    n_episode: scalar, number of episodes
    starting_state: scalar, specifting initial state

    Methods
    ---------
    """

    def __init__(self, model_name="SR", **mod_kwargs):
        """
        Parameters
        ----------
        """
        super().__init__(model_name, **mod_kwargs)
        self.metadata = {"mod_kwargs": mod_kwargs}
        self.obs_history = []  # Initialize observation history to update weights later
        self.gamma = mod_kwargs["discount"]
        self.threshold = mod_kwargs["threshold"]
        self.learning_rate = mod_kwargs["lr_td"]
        self.t_episode = mod_kwargs["t_episode"]
        self.n_episode = mod_kwargs["n_episode"]
        self.reset()

        self.room_width = mod_kwargs["room_width"]
        self.room_depth = mod_kwargs["room_depth"]
        self.state_density = mod_kwargs["state_density"]
        self.resolution_d = int(self.state_density * self.room_depth)
        self.resolution_w = int(self.state_density * self.room_width)
        self.x_array = np.linspace(-self.room_width / 2, self.room_width / 2, num=self.resolution_d)
        self.y_array = np.linspace(self.room_depth / 2, -self.room_depth / 2, num=self.resolution_w)
        self.mesh = np.array(np.meshgrid(self.x_array, self.y_array))
        self.xy_combinations = self.mesh.T.reshape(-1, 2)
        self.w = int(self.room_width * self.state_density)
        self.l = int(self.room_depth * self.state_density)
        self.n_state = int(self.l * self.w)
        self.srmat = []
        self.srmat_sum = []
        self.srmat_ground = []
        self.transmat_norm = []
        self.obs_history = []
        twoD=mod_kwargs['twoD']
        if twoD==True:
            self.create_transmat(self.state_density,  '2D_env')




    def reset(self):
        """
        Initialize
        """
        self.srmat = []
        self.srmat_sum = []
        self.srmat_ground = []
        self.transmat_norm = []
        self.w = 0
        self.l= 0
        self.global_steps = 0  # Reset global steps
        self.obs_history = []  # Reset observation history

    def obs_to_state(self, pos):
        node_layout = np.arange(self.n_state).reshape(self.l, self.w)
        diff = self.xy_combinations - pos[np.newaxis, ...]
        dist = np.sum(diff ** 2, axis=1)
        index = np.argmin(dist)
        curr_state=index
        return curr_state

    def act(self,obs):
        self.obs_history.append(obs)
        if len(self.obs_history) >= 1000:
            self.obs_history = [obs, ]
        arrow=[[0,1],[0,-1],[1,0],[-1,0]]
        action = np.random.normal(scale=0.1, size=(2,))
        diff = action- arrow
        dist = np.sum(diff ** 2, axis=1)
        index = np.argmin(dist)
        action = arrow[index]
        return action

    def get_T_from_M(self, M):
        T=(1/self.gamma)*np.linalg.inv(M)@(M-np.eye(self.n_state))
        return T

    def create_transmat(self, state_density, name_env, plotting_varible=True):
        global transmat_norm

        if name_env == '2D_env':
            '''
            TaskGraph for rectangular environment

            Args:
                param: dictionary {'rec_length': length,'rec_width': width, 'lattice': 'sqaure' / 'tri'}

            Returns:y
                adjmat: adjacency matrix
                node2xy: xy coordinates of each state for plotting (via networkx)
                groups: group labels for each state, always []
            '''



            adjmat_triu = np.zeros((self.n_state, self.n_state))
            node2xy = []
            node_layout = np.arange(self.n_state).reshape(self.l, self.w)
            x_coord = np.linspace(0, np.min([self.l / self.w, 1]), num=self.l)
            y_coord = np.linspace(0, np.min([self.w / self.l, 1]), num=self.w)
            self.xy=[]

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

        if plotting_varible==True:
            f, ax = plt.subplots(1, 1, figsize=(14, 5))
            ax.imshow(self.transmat_norm, interpolation='nearest')

        return self.transmat_norm

    def update_successor_rep(self):
        '''
        Compute the successor representation matrix using geometric sums

        Args:
            transmat: (n_state, n_state) numpy array, transition matrix
            gamma: scalar, discounting factor

        Returns:
            srmat: (n_state, n_state) numpy array, successor representation matrix
        '''
        transmat_type = np.array(self.transmat_norm, dtype=np.float64)

        self.srmat_ground = np.linalg.inv(np.eye(self.n_state) - self.gamma * transmat_type)
        return self.srmat_ground

    def successor_rep_sum(self):
        '''
        Compute the successor representation using successive additive update

        Args:
            transmat: (n_state, n_state) numpy array, transition matrix
            gamma: scalar discount rate
            threoshold: scalar, upper bound for the update size

        Returns:
            srmat: (n_state, n_state) numpy array, successor representation matrix
        '''

        self.srmat_sum = np.zeros_like(self.transmat_norm)
        keep_going = True
        while keep_going:
            new_srmat = self.gamma * self.transmat_norm.dot(self.srmat_sum) + np.eye(self.n_state)
            update = new_srmat - self.srmat_sum
            self.srmat_sum = new_srmat
            if np.max(np.abs(update)) < self.threshold:
                keep_going = False

        return self.srmat_sum

    def update_successor_rep_td(self,obs,curr_state):
        """
        Compute the successor representation matrix using TD learning
        Args:
            transmat: (n_state, n_state) numpy array, transition matrix
        Returns:
            srmat: (n_state, n_state) successor representation matrix
            srmat_snapshots: dictionary, recording srmat at different time steps during learning
        """
        next_state = self.obs_to_state(obs)
        self.n_state = self.transmat_norm.shape[0]
        a = np.array(curr_state)
        x = a.flatten()
        b = np.eye(self.n_state)[x, :self.n_state]
        L = b.reshape(a.shape + (self.n_state,))
        curr_state_vec = L

        self.srmat[:, curr_state] = self.srmat[:, curr_state] + self.learning_rate * (curr_state_vec +
                                                                                self.gamma * self.srmat[:,
                                                                                             next_state] - self.srmat[:,
                                                                                                                  curr_state])

        return next_state,self.srmat

    def update_successor_rep_td_full(self):
        """
        Compute the successor representation matrix using TD learning

        Args:
            transmat: (n_state, n_state) numpy array, transition matrix


        Returns:
            srmat: (n_state, n_state) successor representation matrix
            srmat_snapshots: dictionary, recording srmat at different time steps during learning
        """
        random_state = np.random.RandomState(1234)

        t_elapsed = 0
        srmat0 = np.eye(self.n_state)
        srmat = srmat0.copy()
        for i in range(self.n_episode):

            curr_state = random_state.randint(self.n_state)
            for j in range(self.t_episode):
                a = np.array([curr_state])
                x = a.flatten()
                b = np.eye(self.n_state)[x, :self.n_state]
                L = b.reshape(a.shape + (self.n_state,))

                curr_state_vec = L
                random_state.multinomial(1, self.transmat_norm[curr_state, :])
                next_state = np.where(random_state.multinomial(1, self.transmat_norm[curr_state, :]))[0][0]

                srmat[:, curr_state] = srmat[:, curr_state] + self.learning_rate * (curr_state_vec +
                                                                                    self.gamma * srmat[:,
                                                                                                 next_state] - srmat[:,
                                                                                                               curr_state])
                curr_state = next_state
                t_elapsed += 1

        return srmat

    def plot_trantion(self,matrix, save_path=None, ax=None):
        evals, evecs = np.linalg.eig(matrix)
        if ax is None:
            f, ax = plt.subplots(1,2, figsize=(14, 5))
            ax[0].imshow(self.transmat_norm)
            ax[1].imshow(matrix)
        if not save_path is None:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close("all")
        return ax

    def plot_eigen(self,matrix, save_path, ax=None):
        """"
        Parameters
        ----------
        """
        evals, evecs = np.linalg.eig(matrix)
        if ax is None:
            f, ax = plt.subplots(1, 5, figsize=(14, 5))
            ax[0].imshow(matrix)
            evecs_0 = evecs[:, 1].reshape(self.w, self.l).real
            ax[1].imshow(evecs_0)
            evecs_1 = evecs[:, 2].reshape(self.w, self.l).real
            ax[2].imshow(evecs_1)
            evecs_2 = evecs[:, 3].reshape(self.w, self.l).real
            ax[3].imshow(evecs_2)
            evecs_3 = evecs[:, 5].reshape(self.w, self.l).real
            ax[4].imshow(evecs_3)
            im = ax[4].imshow(evecs_3)
            cbar = plt.colorbar(im, ax=ax[4])

        if not save_path is None:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close("all")
        return ax







