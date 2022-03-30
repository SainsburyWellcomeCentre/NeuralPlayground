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

from models.core import NeuralResponseModel
from environments.environments.simple2d import Simple2D, Sargolini2006, BasicSargolini2006


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

    def obs_to_state(self, obs):
        pos=obs
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
                param: dictionary {'rec_length': length,
                                    'rec_width': width,
                                    'lattice': 'sqaure' / 'tri'}

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

    def update_successor_rep(self, transmat):
        '''
        Compute the successor representation matrix using geometric sums

        Args:
            transmat: (n_state, n_state) numpy array, transition matrix
            gamma: scalar, discounting factor

        Returns:
            srmat: (n_state, n_state) numpy array, successor representation matrix
        '''
        transmat = np.array(transmat, dtype=np.float64)

        self.srmat_ground = np.linalg.inv(np.eye(self.n_state) - self.gamma * transmat)
        return self.srmat_ground

    def successor_rep_sum(self, transmat):
        '''
        Compute the successor representation using successive additive update

        Args:
            transmat: (n_state, n_state) numpy array, transition matrix
            gamma: scalar discount rate
            threoshold: scalar, upper bound for the update size

        Returns:
            srmat: (n_state, n_state) numpy array, successor representation matrix
        '''



        self.srmat_sum = np.zeros_like(transmat)
        keep_going = True
        while keep_going:
            new_srmat = self.gamma * transmat.dot(self.srmat_sum) + np.eye(self.n_state)
            update = new_srmat - self.srmat_sum
            self.srmat_sum = new_srmat
            if np.max(np.abs(update)) < self.threshold:
                keep_going = False

        return self.srmat_sum

    def update_successor_rep_td(self, next_state,curr_state,):
        """
        Compute the successor representation matrix using TD learning
        Args:
            transmat: (n_state, n_state) numpy array, transition matrix
        Returns:
            srmat: (n_state, n_state) successor representation matrix
            srmat_snapshots: dictionary, recording srmat at different time steps during learning
        """
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
        return self.srmat,

    def update_successor_rep_td_full(self, transmat):
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
                random_state.multinomial(1, transmat[curr_state, :])
                next_state = np.where(random_state.multinomial(1, transmat[curr_state, :]))[0][0]

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



if __name__ == "__main__":
    run_raw_data= False

    if run_raw_data== False:
        room_width = 7
        room_depth = 7
        env_name = "env_example"
        time_step_size = 1  # seg
        agent_step_size = 1

        # Init environment
        env = Simple2D(environment_name=env_name,
                       room_width=room_width,
                       room_depth=room_depth,
                       time_step_size=time_step_size,
                       agent_step_size=agent_step_size)

        discount = .9
        threshold = 1e-6
        lr_td = 1e-2
        t_episode = 100
        n_episode = 10000
        state_density = int( 1/ agent_step_size)
        w = env.room_width * state_density
        l = env.room_depth * state_density

        agent = SR(discount=discount, t_episode=t_episode, n_episode=n_episode, threshold=threshold, lr_td=lr_td,
                   room_width=env.room_width, room_depth=env.room_depth,state_density= state_density )
        transmat = agent.create_transmat(state_density,
                                         '2D_env')
        # Choose your function depending on the type of env '2D_env' or '1D_env' + initialisies the smart as well
        sr = agent.update_successor_rep(transmat)  # Choose your type of Update
        sr_td = agent.update_successor_rep_td_full(transmat)  # Choose your type of Update
        sr_sum= agent.successor_rep_sum(transmat)
        agent.plot_eigen(sr, save_path="./figures/ground_truth.pdf")
        agent.plot_eigen(sr_sum, save_path="figures/sr_sum.pdf")
        agent.plot_eigen(sr_td, save_path="./figures/sr_full_td.pdf")

        plot_every = 10000
        total_iters = 0
        obs, state = env.reset()

        current_state=agent.obs_to_state(obs)
        for i in tqdm(range(n_episode)):
            for j in range(t_episode):
                # Observe to choose an action
                action = agent.act(obs)  #the action is link to density of state to make sure we always land in a new
                obs, state, reward = env.step(action)
                new_state = agent.obs_to_state(obs)
                K = agent.update_successor_rep_td(new_state,current_state)
                M_seq=np.asarray(K)
                M_array_sq= M_seq.sum(axis=0)
                current_state=new_state
                total_iters += 1
                if total_iters % plot_every == 0:
                    agent.plot_eigen(M_array_sq, save_path="./figures/M_processed_iter_" + str(total_iters) + ".pdf")
        T= agent.get_T_from_M( M_array_sq)
        agent.plot_trantion(T, save_path="./figures/transtion.pdf")
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

        agent_step_size = 10

        discount = .9
        threshold = 1e-6
        lr_td = 1e-2
        t_episode = 1000
        n_episode = 5000
        state_density = (1 / agent_step_size)
        w = env.room_width * state_density
        l = env.room_depth * state_density

        agent = SR(discount=discount, t_episode=t_episode, n_episode=n_episode, threshold=threshold, lr_td=lr_td,
                   room_width=env.room_width, room_depth=env.room_depth, state_density=state_density)
        transmat = agent.create_transmat(state_density,
                                         '2D_env')
        # Choose your function depending on the type of env '2D_env' or '1D_env' + initialisies the smart as well
        sr = agent.update_successor_rep(transmat)  # Choose your type of Update
        sr_td = agent.update_successor_rep_td_full(transmat)  # Choose your type of Update
        sr_sum = agent.successor_rep_sum(transmat)
        # agent.plot_eigen(sr, save_path="./figures/ground_truth.pdf")
        # agent.plot_eigen(sr_sum, save_path="figures/sr_sum.pdf")
        # agent.plot_eigen(sr_td, save_path="./figures/sr_full_td.pdf")

        plot_every = 1000000
        total_iters = 0
        obs, state = env.reset()

        #for i in tqdm(range(env.total_number_of_steps)):
        obs = obs[:2]
        current_state = agent.obs_to_state(obs)
        for i in tqdm(range(10000000)):
                # Observe to choose an action

                action = agent.act(obs) # the action is link to density of state to make sure we always land in a new
                obs, state, reward = env.step(action)
                obs= obs[:2]
                new_state = agent.obs_to_state(obs)
                K = agent.update_successor_rep_td(new_state, current_state)
                M_seq = np.asarray(K)
                M_array_sq = M_seq.sum(axis=0)
                current_state = new_state
                total_iters += 1
                if total_iters % plot_every == 0:
                    agent.plot_eigen(M_array_sq, save_path="./figures/M_processed_iter_" + str(total_iters) + ".pdf")


