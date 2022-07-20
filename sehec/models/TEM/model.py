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
from sehec.models.TEM.agent_policy import *

fu_co = layer.fully_connected
eps = 1e-8


class TEM(NeuralResponseModel):
    def __init__(self, model_name="TEM", **mod_kwargs):
        super().__init__(model_name, **mod_kwargs)
        self.metadata = {"mod_kwargs": mod_kwargs}
        self.obs_history = []  # Initialize observation history to update weights later
        self.pars = mod_kwargs

        # Inputs for TEM
        self.table, rev_table = combins_table(pars['s_size_comp'], 2)
        self.x_p, self.x_g, self.x_gt = [0] * pars['t_episode'], [0] * pars['t_episode'], [0] * pars['t_episode']
        self.x_ = [0] * pars['t_episode']
        self.g = [0] * pars['t_episode']
        self.no_direction = None

        # Initialisation of Possible Floor Objects
        self.poss_objects = np.zeros(shape=(pars['s_size'], pars['s_size']))
        for i in range(pars['s_size']):
            for j in range(pars['s_size']):
                if j == i:
                    self.poss_objects[i][j] = 1

        self.reset()

    def reset(self):
        # Reset observation history
        self.obs_history = []

    def initialise_hebbian(self):
        # Initialise Hebbian matrices for memory retrieval
        a_rnn = np.zeros((pars['batch_size'], pars['p_size'], pars['p_size']))
        a_rnn_inv = np.zeros((pars['batch_size'], pars['p_size'], pars['p_size']))

        return a_rnn, a_rnn_inv

    def initialise_variables(self):
        # Initialise variables for use in TEM model
        gs = np.maximum(np.random.randn(pars['batch_size'], pars['g_size']) * pars['g_init'], 0)
        x_s = np.zeros((pars['batch_size'], pars['s_size_comp'] * pars['n_freq']))

        n_states = pars['n_states_world']
        visited = np.zeros(pars['batch_size'], max(n_states))  # Used when computing losses

        return gs, x_s, visited

    def obs_to_states(self, pos, batch):
        # Converts position to SR state
        curr_states = []
        room_width = pars['widths'][batch]
        room_depth = pars['widths'][batch]

        resolution_d = int(pars['state_density'] * room_depth)
        resolution_w = int(pars['state_density'] * room_width)
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
        # Produce trajectory of actions from random policy (not used whilst agent_policy.py being used)
        actions = np.zeros((pars['batch_size'], 2, pars['t_episode']))
        direc = np.zeros(shape=(pars['batch_size'], 4, pars['t_episode']))
        for batch in range(pars['batch_size']):
            n_states = pars['widths'][batch] ** 2
            self.obs_history.append(obs)
            if len(self.obs_history) >= 1000:
                self.obs_history = [obs, ]

            arrow = [[0, 1], [0, -1], [1, 0], [-1, 0]]
            batch_action = np.random.normal(scale=0.1, size=(2, 25))
            for step in range(pars['t_episode']):
                diff = batch_action[:, step] - arrow
                dist = np.sum(diff ** 2, axis=1)
                index = np.argmin(dist)
                action = arrow[index]
                actions[batch, :, step] = action
                direc[batch, :, step] = direction(action)

        return actions, direc

    def update(self, direcs, obs, gs, x_s, visited, no_d):
        # Updates all internal representations of TEM
        self.obs_history.append(obs)
        if len(self.obs_history) >= 1000:
            self.obs_history = [obs, ]

        xs = np.zeros(shape=(pars['batch_size'], pars['s_size'], pars['t_episode']))
        self.no_direction = no_d

        for batch in range(pars['batch_size']):
            # Generate landscape of objects in each environment
            objects = np.zeros(shape=(pars['n_states'][batch], pars['s_size']))
            for i in range(pars['n_states'][batch]):
                rand = random.randint(0, pars['s_size'] - 1)
                objects[i] = self.poss_objects[rand]

            # Make observations of sensorium in SR states
            for step in range(pars['t_episode']):
                state = self.obs_to_states(obs[batch, :, step], batch)
                observation = objects[state]
                xs[batch, :, step] = observation

        for i in range(pars['t_episode']):
            self.seq_pos = 1 * pars['t_episode'] + i
            # Initialisation of filtered x and g
            if i == 0:
                x_s = tf.split(axis=1, num_or_size_splits=pars['n_freq'], value=x_s)
                x_t = x_s
                g_t = gs
            # Pointing to previous x and g
            else:
                x_t = self.x_[i - 1]
                g_t = self.g[i - 1]

            # Two-hot Encoding
            x_two_hot = onehot2twohot(self, xs, self.table, pars['s_size_comp'])

            # Temporally filter
            x_ = self.x2x_(x_two_hot, x_t)

            # Generative Transition
            g_gen, g2g_all = self.gen_g(g_t, direcs[:, :, i])

            # Update internal representations
            self.x_[i] = x_
            self.g[i] = g_gen

            print("finished step ", i)

        return

    # HELPER FUNCTIONS
    def hierarchical_logsig(self, x, name, splits, sizes, trainable, concat, k=2):
        xs = x if splits == 'done' else tf.split(value=x, num_or_size_splits=splits, axis=1)
        xs = [tf.stop_gradient(x) for x in xs]

        logsigs_ = [fu_co(xs[i], k * sizes[i], activation_fn=tf.nn.elu, reuse=tf.AUTO_REUSE, scope=name + '_' + str(i),
                          weights_initializer=layer.xavier_initializer(),
                          trainable=trainable) for i in range(pars['n_freq'])]
        logsigs = [pars['logsig_ratio'] * fu_co(logsigs_[i], sizes[i], activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE,
                                                scope=name + str(i), weights_initializer=layer.xavier_initializer(),
                                                trainable=trainable) for i in range(pars['n_freq'])]

        return tf.concat(logsigs, axis=1) if concat else logsigs

    def x2x_(self, x, x_):
        # Temporally filter sensorium (into 5 frequencies)
        x_ = [0] * pars['n_freq']
        for i in range(pars['n_freq']):
            with tf.variable_scope("x2x_" + str(i), reuse=tf.AUTO_REUSE):
                gamma = tf.get_variable("w_smooth_freq", [1], initializer=tf.constant_initializer(
                    np.log(pars['freq'][i] / (1 - pars['freq'][i]))),
                                        trainable=True)
            # Inverse sigmoid as initial parameter
            a = tf.sigmoid(gamma)
            # Filter
            x_[i] = a * x_[i] + x * (1 - a)

        return x_

    def gen_g(self, g, d):
        # Generative prior on grids if first step in environment, else transition
        g, sigma = tf.cond(tf.constant(self.seq_pos > 0, dtype=tf.bool),
                            true_fn=lambda: self.g2g(g, d, pars['no_direc_gen'], name='gen'),
                            false_fn=lambda: self.g_prior())
        # Same but for used for inference network
        g_inf, sigma_inf = tf.cond(tf.constant(self.seq_pos > 0, dtype=tf.bool),
                                    true_fn=lambda: self.g2g(g, d, False, name='inf'),
                                    false_fn=lambda: self.g_prior())

        return g, (g_inf, sigma_inf)

    def g2g(self, g_old, d, no_direc=False, name=''):
        # Make grid to grid transition
        # Transition update
        update = self.get_g2g_update(g_old, d, no_direc, name='')
        # Add on update to current representation
        g = update + g_old
        # Apply activation
        g = self.f_g(g)
        # Get variance
        logsig = self.hierarchical_logsig(g, 'sig_g2g' + name, self.pars['n_grids_all'], self.pars['n_grids_all'],
                                          self.pars['train_sig_g2g'], concat=True)
        logsig += self.pars['logsig_offset']

        sigma = tf.exp(logsig)

        return g, sigma

    def get_g2g_update(self, g_p, d, no_direc, name=''):
        # Calculate update to EC representation
        # Get transition matrix
        t_mat = self.get_transition(d, name)
        # multiply current entorhinal representation by transition matrix
        update = tf.reshape(tf.matmul(t_mat, tf.reshape(g_p, [pars['batch_size'], pars['g_size'], 1])),
                            [pars['batch_size'], pars['g_size']])

        if no_direc:
            # directionless transition weights - used in OVC environments
            with tf.variable_scope("g2g_directionless_weights" + name, reuse=tf.AUTO_REUSE):
                t_mat_2 = tf.get_variable("g2g" + name, [pars['g_size'], pars['g_size']])
                t_mat_2 = tf.multiply(t_mat_2, self.pars['mask_g'])

            update = tf.where(self.no_direction > 0.5, x=tf.matmul(g_p, t_mat_2), y=update)

        return update

    def g_prior(self, name=''):
        # Initial EC representation
        with tf.variable_scope("g_prior", reuse=tf.AUTO_REUSE):
            g = tf.tile(tf.get_variable("mu_g_prior" + name, [1, pars['g_size']],
                                         initializer=tf.truncated_normal_initializer(stddev=pars['g_init'])),
                         [pars['batch_size'], 1])
            logsig = tf.tile(tf.get_variable("logsig_g_prior" + name, [1, pars['g_size']],
                                             initializer=tf.truncated_normal_initializer(stddev=pars['g_init'])
                                             ), [pars['batch_size'], 1])

        sigma = tf.exp(logsig)

        return tf.cast(g, tf.float64), tf.cast(sigma, tf.float64)

    def get_transition(self, d, name=''):
        # Get transition matrix based on relationship / action
        d_mixed = fu_co(d, pars['d_mixed_size'], activation_fn=tf.tanh, reuse=tf.AUTO_REUSE,
                        scope='d_mixed_g2g' + name) if pars['d_mixed'] else d

        t_vec = tf.layers.dense(d_mixed, pars['g_size'] ** 2, activation=None, reuse=tf.AUTO_REUSE,
                                name='mu_g2g' + name, kernel_initializer=tf.zeros_initializer, use_bias=False)
        # Turn vector into matrix
        trans_all = tf.reshape(t_vec, [pars['batch_size'], pars['g_size'], pars['g_size']])
        # Apply mask - i.e. if hierarchically or only transition within frequency
        trans_all = tf.multiply(trans_all, self.pars['mask_g'])

        return trans_all

    def f_g(self, g):
        # Apply activation to EC representation
        with tf.name_scope('f_g'):
            gs = tf.split(value=g, num_or_size_splits=self.pars['n_grids_all'], axis=1)
            for i in range(self.pars['n_freq']):
                # Apply activation to each frequency separately
                gs[i] = self.f_g_freq(gs[i], i)

            g = tf.concat(gs, axis=1)

        return g

    def f_g_freq(self, g, freq):
        g = self.pars['g2g_activation'](g)

        return g


def curriculum(pars_orig, pars, n_restart):
    n_envs = len(pars['widths'])
    b_s = int(pars['batch_size'])
    # Choose pars for current stage of training
    rn = np.random.randint(low=-pars['seq_jitter'], high=pars['seq_jitter'])
    n_restart = np.maximum(n_restart - pars['curriculum_steps'], pars['restart_min'])

    pars['direc_bias_env'] = [0 for _ in range(n_envs)]

    # Make choice for each env
    choices = []
    for env in range(n_envs):
        choice = np.random.choice(pars['poss_behaviours'])

        choices.append(choice)

        if choice == 'normal':
            pars['direc_bias_env'][env] = pars_orig['direc_bias']
        else:
            raise Exception('Not a correct possible behaviour')

    # Choose which of batch gets no_direc or not - 1 is no_direc, 0 is with direc
    no_direc_batch = np.ones(pars['batch_size'])
    for batch in range(b_s):
        env = pars['diff_env_batches_envs'][batch]
        choice = choices[env]
        if choice == 'normal':
            no_direc_batch[batch] = 0
        else:
            no_direc_batch[batch] = 1

    return pars, rn, n_restart, no_direc_batch


def direction(action):
    # Turns action [x,y] into direction [R,L,U,D]
    x, y = action
    direc = np.zeros(shape=4)
    if x > 0 and y == 0:
        d = 0
        name = 'right'
        direc[d] = 1
    elif x < 0 and y == 0:
        d = 1
        name = 'left'
        direc[d] = 1
    elif x == 0 and y > 0:
        d = 2
        name = 'up'
        direc[d] = 1
    elif x == 0 and y < 0:
        d = 3
        name = 'down'
        direc[d] = 1
    else:
        ValueError('impossible action')

    return direc


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
    # Produces a table of s_size two-hot encoded vectors, each of size 10
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
    # Compresses one-hot vector of size 45 to two-hot of size 10
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
pars = default_params()
pars_orig = pars.copy()

env_name = "TEMenv"
mod_name = "TEM"
n_restart = pars['restart_max'] + pars['curriculum_steps']

# Initialise Environment(s)
envs = TEMenv(environment_name=env_name, **pars)
agent = TEM(model_name=mod_name, **pars)

# Curriculum of behaviour types
pars, rn, n_restart, no_direc_batch = curriculum(pars_orig, pars, n_restart)

# Initialise Environment and Variables (same each batch)
gs, x_s, visited = agent.initialise_variables()

for i in range(pars['n_episode']):
    obs, state = envs.reset()

    # Initalise Hebbian Weights
    a_rnn, a_rnn_inv = agent.initialise_hebbian()

    # RL Loop
    # actions, direc = act(obs)
    obs, states, rewards, actions, direcs = envs.step(obs)
    xs = obs
    agent.update(direcs, obs, gs, x_s, visited, no_direc_batch)

envs.plot_trajectory()
plt.show()
