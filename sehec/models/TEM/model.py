import sys

sys.path.append("../")
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.layers as layer
from scipy import special
from tqdm import tqdm

from sehec.models.modelcore import NeuralResponseModel
from sehec.envs.arenas.TEMenv import TEMenv
from sehec.models.TEM.parameters import *

fu_co = layer.fully_connected
eps = 1e-8


class TEM(NeuralResponseModel):
    def __init__(self, model_name="TEM", **mod_kwargs):
        super().__init__(model_name, **mod_kwargs)
        self.metadata = {"mod_kwargs": mod_kwargs}
        self.obs_history = []  # Initialize observation history to update weights later
        self.world_type = mod_kwargs['world_type']
        self.gamma = mod_kwargs["discount"]
        self.threshold = mod_kwargs["threshold"]
        self.learning_rate = mod_kwargs["lr_td"]
        self.t_episode = mod_kwargs["t_episode"]
        self.widths = mod_kwargs["widths"]
        self.state_density = mod_kwargs["state_density"]
        self.n_states_world = mod_kwargs["n_states_world"]
        twoD = mod_kwargs['twoDvalue']
        self.inital_obs_variable = None

        self.reset()

        # Model Parameters
        self.infer_g_type = 'g_p'  # 'g'
        self.no_direc_gen = False

        # Inputs for TEM
        self.batch_size = 16
        self.n_phases_all = [10, 10, 8, 6, 6]
        self.freq = [0.01, 0.7, 0.91, 0.97, 0.99, 0.9995]
        self.s_size = 45
        self.s_size_comp = 10
        self.g_init = 0.5
        self.logsig_ratio = 6
        self.tot_phases = sum(self.n_phases_all)
        self.n_freq = len(self.n_phases_all)
        self.table, rev_table = combins_table(self.s_size_comp, 2)
        self.n_grids_all = [int(3 * n_phase) for n_phase in self.n_phases_all]
        # self.x_ = np.zeros(shape=(self.s_size_comp * self.n_freq))
        self.x_p, self.x_g, self.x_gt = [0] * self.t_episode, [0] * self.t_episode, [0] * self.t_episode
        self.x_ = [0] * self.t_episode
        self.g = [0] * self.t_episode
        self.g_size = sum(self.n_grids_all)
        self.p_size = int(self.tot_phases * self.s_size_comp)

        self.d_mixed = True
        self.d_mixed_size = 15 if self.world_type == 'square' else 20

        self.obs_history = []

    def reset(self):
        self.srmat = []
        self.srmat_sum = []
        self.srmat_ground = []
        self.transmat_norm = []
        self.inital_obs_variable = None
        self.obs_history = []  # Reset observation history

    def obs_to_states(self, pos):
        curr_states = []
        for i in range(self.batch_size):
            room_width = self.widths[i]
            room_depth = room_width
            resolution_d = int(self.state_density * room_depth)
            resolution_w = int(self.state_density * room_width)
            x_array = np.linspace(-room_width / 2, room_width / 2, num=resolution_d)
            y_array = np.linspace(room_depth / 2, -room_depth / 2, num=resolution_w)
            mesh = np.array(np.meshgrid(x_array, y_array))
            xy_combinations = mesh.T.reshape(-1, 2)

            diff = xy_combinations - pos[np.newaxis, ...]
            dist = np.sum(diff ** 2, axis=1)
            index = np.argmin(dist)
            curr_state = index
            curr_states.append(curr_state)

        return curr_states

    def act(self, obs):
        actions = np.zeros((self.batch_size, 2, self.t_episode))
        xs = np.zeros((self.batch_size, self.s_size, self.t_episode))

        for batch in range(self.batch_size):
            n_states = self.widths[batch] ** 2
            self.obs_history.append(obs)
            if len(self.obs_history) >= 1000:
                self.obs_history = [obs, ]

            arrow = [[0, 1], [0, -1], [1, 0], [-1, 0]]
            batch_action = np.random.normal(scale=0.1, size=(2, 25))
            for step in range(self.t_episode):
                diff = batch_action[:, step] - arrow
                dist = np.sum(diff ** 2, axis=1)
                index = np.argmin(dist)
                action = arrow[index]
                actions[batch, :, step] = action

            # next_states = self.obs_to_states(obs)

            # xs[:, :, step] = objects[next_states]
            # x_, x_two_hot = calculate(self, xs)

        return actions

    def initialise_hebbian(self):
        a_rnn = np.zeros((self.batch_size, self.p_size, self.p_size))
        a_rnn_inv = np.zeros((self.batch_size, self.p_size, self.p_size))

        return a_rnn, a_rnn_inv

    def initialise_variables(self):
        gs = np.maximum(np.random.randn(self.batch_size, self.g_size) * self.g_init, 0)
        x_s = np.zeros((self.batch_size, self.s_size_comp * self.n_freq))

        n_states = self.n_states_world
        visited = np.zeros(self.batch_size, max(n_states))

        return gs, x_s, visited

    def update(self, actions, x, x_s, gs):
        for i in range(self.t_episode):
            self.seq_pos = 1 * self.t_episode + i
            # Initialisation of filtered x
            if len(self.obs_history) == 1:
                x_s = tf.split(axis=1, num_or_size_splits=self.n_freq, value=x_s)
                x_t = x_s
                g_t = gs
            # Pointing to previous
            else:
                x_t = self.x_[i - 1]
                g_t = self.g[i - 1]

            # Two-hot Encoding
            x_two_hot = onehot2twohot(self, x, self.table, self.s_size_comp)

            # Temporally filter
            x_ = self.x2x_(x_two_hot, x_t)

            # Generative Transition
            g_gen, g2g_all = self.gen_g(g_t, actions)

        return x_, x_two_hot

    # HELPER FUNCTIONS
    def hierarchical_logsig(self, x, name, splits, sizes, trainable, concat, k=2):
        xs = x if splits == 'done' else tf.split(value=x, num_or_size_splits=splits, axis=1)
        xs = [tf.stop_gradient(x) for x in xs]

        logsigs_ = [fu_co(xs[i], k * sizes[i], activation_fn=tf.nn.elu, reuse=tf.AUTO_REUSE, scope=name + '_' + str(i),
                          weights_initializer=layer.xavier_initializer(),
                          trainable=trainable) for i in range(self.n_freq)]
        logsigs = [self.logsig_ratio * fu_co(logsigs_[i], sizes[i], activation_fn=tf.nn.tanh,
                                             reuse=tf.AUTO_REUSE, scope=name + str(i),
                                             weights_initializer=layer.xavier_initializer(),
                                             trainable=trainable) for i in range(self.n_freq)]

        return tf.concat(logsigs, axis=1) if concat else logsigs

    def x2x_(self, x, x_):
        x_ = [0] * self.n_freq
        for i in range(self.n_freq):
            with tf.variable_scope("x2x_" + str(i), reuse=tf.AUTO_REUSE):
                gamma = tf.get_variable("w_smooth_freq", [1], initializer=tf.constant_initializer(
                    np.log(self.freq[i] / (1 - self.freq[i]))),
                                        trainable=True)
            # Inverse sigmoid as initial parameter
            a = tf.sigmoid(gamma)
            # Filter
            x_[i] = a * x_[i] + x * (1 - a)

        return x_

    def gen_g(self, g, d):
        """generates grid cells from previous time step - sepatated into when for inferene and generation"""
        # generative prior on grids if first step in environment, else transition
        mu = tf.cond(tf.constant(self.seq_pos > 0, dtype=tf.bool),
                     true_fn=lambda: self.g2g(g, d, self.no_direc_gen, name='gen'),
                     false_fn=lambda: self.g_prior())

        # the same but for used for inference network
        mu_inf = tf.cond(tf.constant(self.seq_pos > 0, dtype=tf.bool),
                         true_fn=lambda: self.g2g(g, d, False, name='inf'),
                         false_fn=lambda: self.g_prior())

        return mu, mu_inf

    def g2g(self, g, d, no_direc=False, name=''):
        """make grid to grid transisiton"""
        # transition update
        update = self.get_g2g_update(g, d, no_direc, name='')
        # add on update to current representation
        mu = update + g
        # apply activation
        mu = self.f_g(mu)

        return mu

    def get_g2g_update(self, g_p, d, no_direc, name=''):
        # get transition matrix
        t_mat = self.get_transition(d, name)
        # multiply current entorhinal representation by transition matrix
        update = tf.reshape(tf.matmul(t_mat, tf.reshape(g_p, [self.par['batch_size'], self.par['g_size'], 1])),
                            [self.par['batch_size'], self.par['g_size']])

        if no_direc:
            # directionless transition weights - used in OVC environments
            with tf.variable_scope("g2g_directionless_weights" + name, reuse=tf.AUTO_REUSE):
                t_mat_2 = tf.get_variable("g2g" + name, [self.g_size, self.g_size])
                t_mat_2 = tf.multiply(t_mat_2, self.mask_g)

            update = tf.where(self.no_direction > 0.5, x=tf.matmul(g_p, t_mat_2), y=update)

        return update

    def g_prior(self, name=''):
        """Gives prior distribution for grid cells"""
        with tf.variable_scope("g_prior", reuse=tf.AUTO_REUSE):
            mu = tf.tile(tf.get_variable("mu_g_prior" + name, [1, self.g_size],
                                         initializer=tf.truncated_normal_initializer(stddev=self.g_init)),
                         [self.par['batch_size'], 1])

        return mu

    def get_transition(self, d, name=''):
        # get transition matrix based on relationship / action
        d_mixed = fu_co(d, (self.d_mixed_size), activation_fn=tf.tanh, reuse=tf.AUTO_REUSE,
                        scope='d_mixed_g2g' + name) if self.d_mixed else d

        t_vec = tf.layers.dense(d_mixed, self.g_size ** 2, activation=None, reuse=tf.AUTO_REUSE,
                                name='mu_g2g' + name, kernel_initializer=tf.zeros_initializer, use_bias=False)
        # turn vector into matrix
        trans_all = tf.reshape(t_vec, [self.batch_size, self.g_size, self.g_size])
        # apply mask - i.e. if hierarchically or only transition within frequency
        trans_all = tf.multiply(trans_all, self.mask_g)

        return trans_all


def combins(n, k, m):
    s = []
    for i in range(1, n + 1):
        c = special.comb(n - i, k)
        if m >= c:
            s.append(1)
            m -= c
            k = k - 1
        else:
            s.append(0)
    return tuple(s)


def combins_table(n, k, map_max=None):
    " Produces a table of s_size two-hot encoded vectors, each of size 10."
    table = []
    rev_table = {}
    table_top = special.comb(n, k)

    for m in range(int(table_top)):
        c = combins(n, k, m)
        if map_max is None or m < map_max:
            table.append(c)
            rev_table[c] = m
        else:
            rev_table[c] = m % map_max

    return table, rev_table


def onehot2twohot(self, onehot, table, compress_size):
    seq_len = np.shape(onehot)[2]
    batch_size = np.shape(onehot)[0]
    twohot = np.zeros((batch_size, compress_size, seq_len))
    for i in range(seq_len):
        vals = np.argmax(onehot[:, :, i], 1)
        for b in range(batch_size):
            twohot[b, :, i] = table[vals[int(b)]]

    return twohot


# ------------------------------------------------------------------------------------------------------
# MAIN
env_name = "TEMenv"
mod_name = "TEM"
pars = default_params()
# Initialise Environment(s)
envs = TEMenv(environment_name=env_name, **pars)

agent = TEM(model_name=mod_name, **pars)

for i in range(pars['n_episode']):
    obs, state = envs.reset()
    # obs = obs[:2]

    # Initialise Environment, Weight and Variable Batch
    adjs, trans = envs.make_environment()
    a_rnn, a_rnn_inv = agent.initialise_hebbian()
    gs, x_s, visited = agent.initialise_variables()

    actions = agent.act(obs)
    obs, states, rewards = envs.step(actions)
    # obs = obs[:2]
    xs = obs
    x_, x_two_hot = agent.update(actions, xs, x_s, gs)

# print(np.shape(x), x_, np.shape(x_two_hot))
envs.plot_trajectory()
plt.show()