import sys
sys.path.append("../")

import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy
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

        # Inputs for TEM
        self.objects = np.zeros(shape=(45, self.t_episode))
        for i in range(self.t_episode):
            rand = random.randrange(0, 45)
            self.objects[rand][i] = 1 
        random.shuffle(self.objects[0])

        self. n_phases_all = [10, 10, 8, 6, 6]
        self.n_freq = len(self.n_phases_all)
        self.s_size_comp = 10
        self.table = combins_table(self.s_size_comp, 2)
        self.n_grids_all = [int(3 * n_phase) for n_phase in self.n_phases_all]
        self.g_size = sum(self.n_grids_all)
        self.g_init = 0.5
        
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
            adjs, trans = square_world(self.w)
            # self.create_transmat(self.state_density,  '2D_env')

        # Initialise Variables for TEM
        self.g, self.x_, visited = initialise_variables(self)

        # Print Testing
        print("Transition matrix: ")
        print(trans)
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
        x = self.objects[:,index]

        return curr_state, x

    def act(self, obs):
        self.obs_history.append(obs)
        if len(self.obs_history) >= 1000:
            self.obs_history = [obs, ]
        # # Initialisation of filtered x
        # if len(self.obs_history) == 1:
        #     x_t = np.zeros(shape=(self.s_size_comp * self.n_freq))
        # # Pointing to previous x
        # else:
        #     x_t = self.objects[len(self.obs_history)]
        arrow=[[0,1],[0,-1],[1,0],[-1,0]]
        action = np.random.normal(scale=0.1, size=(2,))
        diff = action- arrow
        dist = np.sum(diff ** 2, axis=1)
        index = np.argmin(dist)
        action = arrow[index]
        self.next_state, self.next_object = self.obs_to_state(obs)

        # # Two-hot Encoding
        # x_two_hot = onehot2twohot(self, self.next_object, self.table, self.s_size_comp)

        # # Temporally filter
        # x_ = x2x_(self, x_two_hot, x_t)

        return action, self.next_object, # x_, x_two_hot


# HELPER FUNCTIONS
def initialise_variables(self):
        gs = np.maximum(np.random.randn(self.g_size) * self.g_init, 0)
        x_s = np.zeros((self.s_size_comp * self.n_freq))
        
        visited = np.zeros(self.n_state)

        return gs, x_s, visited

def square_world(width):
    stay_still = True
    states = int(width ** 2)
    adj = np.zeros((states, states))

    for i in range(states):
        # stay still
        if stay_still:
            adj[i, i] = 1
        # up - down
        if i + width < states:
            adj[i, i + width] = 1
            adj[i + width, i] = 1
            # left - right
        if np.mod(i, width) != 0:
            adj[i, i - 1] = 1
            adj[i - 1, i] = 1

    tran = np.zeros((states, states))
    for i in range(states):
        if sum(adj[i]) > 0:
            tran[i] = adj[i] / sum(adj[i])
    
    f, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.imshow(tran, interpolation='nearest')

    f, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.imshow(adj, interpolation='nearest')

    return adj, tran

def combins(n, k, m):
    s = []
    for i in range(1, n + 1):
        c = scipy.special.comb(n - i, k)
        if m >= c:
            s.append(1)
            m -= c
            k = k - 1
        else:
            s.append(0)
    return tuple(s)

def combins_table(n, k, map_max=None):
    table = []
    rev_table = {}
    table_top = scipy.special.comb(n, k)

    for m in range(int(table_top)):
        c = combins(n, k, m)
        if map_max is None or m < map_max:
            table.append(c)
            rev_table[c] = m
        else:
            rev_table[c] = m % map_max
    
    return table, 
    
def onehot2twohot(self, onehot, table, compress_size):
    seq_len = np.shape(onehot)[2]
    batch_size = np.shape(onehot)[0]
    twohot = np.zeros((batch_size, compress_size, seq_len))
    for i in range(np.shape(onehot)[2]):
        vals = np.argmax(onehot[:, :, i], 1)
        for b in range(np.shape(onehot)[0]):
            twohot[b, :, i] = table[vals[int(b)]]

    return twohot

def x2x_(self, x, x_):
    x_ = [0] * self.n_freq
    for i in range(self.n_freq):
        with tf.variable_scope("x2x_" + str(i), reuse=tf.AUTO_REUSE):
                gamma = tf.get_variable("w_smooth_freq", [1], initializer=tf.constant_initializer(
                    np.log(self.par['freqs'][i] / (1 - self.par['freqs'][i]))),
                                        trainable=True)
        # Inverse sigmoid as initial parameter
        a = tf.sigmoid(gamma)
        # Filter
        x_[i] = a * x_[i] + x * (1 - a)
    
    return x_