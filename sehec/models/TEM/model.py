import sys
sys.path.append("../")

import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

from sehec.models.modelcore import NeuralResponseModel
from sehec.envs.arenas.simple2d import Simple2D


class TEM(NeuralResponseModel):
    def __init__(self, model_name="TEM", **mod_kwargs):
        super().__init__(model_name, **mod_kwargs)
        self.metadata = {"mod_kwargs": mod_kwargs}
        self.obs_history = []  # Initialize observation history to update weights later
        self.gamma = mod_kwargs["discount"]
        self.threshold = mod_kwargs["threshold"]
        self.learning_rate = mod_kwargs["lr_td"]
        self.t_episode = mod_kwargs["t_episode"]
        self.n_episode = mod_kwargs["n_episode"]
        self.room_width = mod_kwargs["room_width"]
        self.room_depth = mod_kwargs["room_depth"]
        self.state_density = mod_kwargs["state_density"]
        twoD = mod_kwargs['twoD']
        self.inital_obs_variable = None

        self.reset()

        # Variables for TEM
        self.objects = np.zeros(shape=(45, self.t_episode))
        
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
        if twoD==True:
            self.create_transmat(self.state_density,  '2D_env')
        for i in range(self.t_episode):
            rand = random.randrange(0, 45)
            self.objects[rand][i] = 1 
        random.shuffle(self.objects[0])

        # Print Testing
        print("n_states: ", self.n_state)
        print("First sensory observation: ", self.objects[:,0])

    def reset(self):
        self.srmat = []
        self.srmat_sum = []
        self.srmat_ground = []
        self.transmat_norm = []
        self.inital_obs_variable = None
        self.obs_history = []  # Reset observation history

    def obs_to_state(self, pos):
        diff = self.xy_combinations - pos[np.newaxis, ...]
        dist = np.sum(diff ** 2, axis=1)
        index = np.argmin(dist)
        curr_state=index
        curr_object = self.objects[:,index]

        return curr_state, curr_object

    def act(self, obs):
        self.obs_history.append(obs)
        if len(self.obs_history) >= 1000:
            self.obs_history = [obs, ]
        arrow=[[0,1],[0,-1],[1,0],[-1,0]]
        action = np.random.normal(scale=0.1, size=(2,))
        diff = action- arrow
        dist = np.sum(diff ** 2, axis=1)
        index = np.argmin(dist)
        action = arrow[index]
        self.next_state, self.next_object = self.obs_to_state(obs)

        return action, self.next_object

    def create_transmat(self, state_density, name_env, plotting_variable=True):
        if name_env == '2D_env':

            adjmat_triu = np.zeros((self.n_state, self.n_state))
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

        if plotting_variable==True:
            f, ax = plt.subplots(1, 1, figsize=(14, 5))
            ax.imshow(self.transmat_norm, interpolation='nearest')

        return self.transmat_norm