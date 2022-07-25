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
from sehec.models.TEM.main import *

fu_co = layer.fully_connected
eps = 1e-8


class TEM(NeuralResponseModel):
    def __init__(self, model_name="TEM", **mod_kwargs):
        super().__init__(model_name, **mod_kwargs)
        self.metadata = {"mod_kwargs": mod_kwargs}
        self.obs_history = []  # Initialize observation history to update weights later
        self.pars = mod_kwargs
        self.pars_orig = self.pars.copy()

        # Variables that are initialised in update( )
        self.temp = self.forget = self.h_l = self.p2g_use = None
        self.g_cell_reg = self.p_cell_reg = self.ovc_cell_reg = None
        self.A = self.A_inv = self.A_split = None
        self.seq_pos = None

        # Information on direction
        self.pars, self.rn, self.n_restart, self.no_direc_batch = direction_pars(self.pars_orig, self.pars, self.pars['n_restart'])

        # Initialise Environment and Variables (same each batch)
        self.gs, self.x_s, self.visited = self.initialise_variables()

        # Inputs for TEM
        self.table, rev_table = combins_table(self.pars['s_size_comp'], 2)
        self.x_p, self.x_g, self.x_gt = [0] * self.pars['t_episode'], [0] * self.pars['t_episode'], [0] * self.pars[
            't_episode']
        self.x_ = [0] * self.pars['t_episode']
        self.g = [0] * self.pars['t_episode']
        self.no_direction = None

        # Initialisation of Possible Floor Objects
        self.poss_objects = np.zeros(shape=(self.pars['s_size'], self.pars['s_size']))
        for i in range(self.pars['s_size']):
            for j in range(self.pars['s_size']):
                if j == i:
                    self.poss_objects[i][j] = 1

        # Memories 
        self.mem_list_a, self.mem_list_b, self.mem_list_e, self.mem_list_f = [], [], [], []
        self.mem_a = tf.zeros([self.pars['batch_size'], self.pars['p_size'], 1], dtype=tf.float32)
        self.mem_b = tf.zeros([self.pars['batch_size'], self.pars['p_size'], 1], dtype=tf.float32)
        self.mem_e = tf.zeros([self.pars['batch_size'], self.pars['p_size'], 1], dtype=tf.float32)
        self.mem_f = tf.zeros([self.pars['batch_size'], self.pars['p_size'], 1], dtype=tf.float32)
        self.mem_list_a_s = [[] for _ in range(self.pars['n_freq'])]
        self.mem_list_b_s = [[] for _ in range(self.pars['n_freq'])]
        self.mem_list_e_s = [[] for _ in range(self.pars['n_freq'])]
        self.mem_list_f_s = [[] for _ in range(self.pars['n_freq'])]
        self.mem_a_s = [tf.zeros([self.pars['batch_size'], self.pars['n_place_all'][i], 1],
                                 dtype=tf.float32) for i in range(self.pars['n_freq'])]
        self.mem_b_s = [tf.zeros([self.pars['batch_size'], self.pars['n_place_all'][i], 1],
                                 dtype=tf.float32) for i in range(self.pars['n_freq'])]
        self.mem_e_s = [tf.zeros([self.pars['batch_size'], self.pars['n_place_all'][i], 1],
                                 dtype=tf.float32) for i in range(self.pars['n_freq'])]
        self.mem_f_s = [tf.zeros([self.pars['batch_size'], self.pars['n_place_all'][i], 1],
                                 dtype=tf.float32) for i in range(self.pars['n_freq'])]

        self.reset()

    def reset(self):
        # Reset observation history
        self.obs_history = []

    def initialise_hebbian(self):
        # Initialise Hebbian matrices for memory retrieval
        a_rnn = np.zeros((self.pars['batch_size'], self.pars['p_size'], self.pars['p_size']))
        a_rnn_inv = np.zeros((self.pars['batch_size'], self.pars['p_size'], self.pars['p_size']))

        return a_rnn, a_rnn_inv

    def initialise_variables(self):
        # Initialise variables for use in TEM model
        gs = np.maximum(np.random.randn(self.pars['batch_size'], self.pars['g_size']) * self.pars['g_init'], 0)
        x_s = np.zeros((self.pars['batch_size'], self.pars['s_size_comp'] * self.pars['n_freq']))

        n_states = self.pars['n_states_world']
        visited = np.zeros(self.pars['batch_size'], max(n_states))  # Used when computing losses

        return gs, x_s, visited

    def obs_to_states(self, pos, batch):
        # Converts position to SR state
        curr_states = []
        room_width = self.pars['widths'][batch]
        room_depth = self.pars['widths'][batch]

        resolution_d = int(self.pars['state_density'] * room_depth)
        resolution_w = int(self.pars['state_density'] * room_width)
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
        action = np.random.normal(scale=0.1, size=2)
        arrow = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        diff = action - arrow
        dist = np.sum(diff ** 2, axis=1)
        index = np.argmin(dist)
        action = arrow[index]
        direc = direction(action)

        return action, direc

    def update(self, direcs, obs):
        # Updates all internal representations of TEM
        self.obs_history.append(obs)
        if len(self.obs_history) >= 1000:
            self.obs_history = [obs, ]

        # Initalise Hebbian matrices each batch
        self.A, self.A_inv = self.initialise_hebbian()
        # split into frequencies for hierarchical attractor - i.e. finish attractor early for low freq memories
        self.A_split = tf.split(self.A, num_or_size_splits=self.pars['n_place_all'], axis=2)

        xs = np.zeros(shape=(self.pars['batch_size'], self.pars['s_size'], self.pars['t_episode']))
        x_s = tf.split(axis=1, num_or_size_splits=self.pars['n_freq'], value=self.x_s)
        self.no_direction = self.no_direc_batch

        for batch in range(self.pars['batch_size']):
            # Generate landscape of objects in each environment
            objects = np.zeros(shape=(self.pars['n_states'][batch], self.pars['s_size']))
            for i in range(self.pars['n_states'][batch]):
                rand = random.randint(0, self.pars['s_size'] - 1)
                objects[i] = self.poss_objects[rand]

            # Make observations of sensorium in SR states
            for step in range(self.pars['t_episode']):
                state = self.obs_to_states(obs[batch, :, step], batch)
                observation = objects[state]
                xs[batch, :, step] = observation

        for i in range(self.pars['t_episode']):
            # get all scaling parameters - they slowly change to help network learn
            self.temp, self.forget, self.h_l, self.p2g_use, l_r, self.g_cell_reg, self.p_cell_reg, self.ovc_cell_reg \
                = self.get_scaling_parameters(i)

            self.seq_pos = 1 * self.pars['t_episode'] + i
            # Initialisation of filtered x and g
            if i == 0:
                x_t = x_s
                g_t = self.gs
            # Pointing to previous x and g
            else:
                x_t = self.x_[i - 1]
                g_t = self.g[i - 1]

            # Two-hot Encoding
            x_two_hot = onehot2twohot(self, xs, self.table, self.pars['s_size_comp'])

            # Generative Transition
            g_gen, g2g_all = self.gen_g(g_t, direcs[:, :, i])

            # Inference
            # self.inference(g2g_all, xs[:, :, i], x_two_hot[:, :, i], x_t)

            # Update internal representations
            # self.x_[i] = x_
            self.g[i] = g_gen

            print("finished step ", i)

        return

    # HELPER FUNCTIONS
    def hierarchical_logsig(self, x, name, splits, sizes, trainable, concat, k=2):
        xs = x if splits == 'done' else tf.split(value=x, num_or_size_splits=splits, axis=1)
        xs = [tf.stop_gradient(x) for x in xs]

        logsigs_ = [fu_co(xs[i], k * sizes[i], activation_fn=tf.nn.elu, reuse=tf.AUTO_REUSE, scope=name + '_' + str(i),
                          weights_initializer=layer.xavier_initializer(),
                          trainable=trainable) for i in range(self.pars['n_freq'])]
        logsigs = [
            self.pars['logsig_ratio'] * fu_co(logsigs_[i], sizes[i], activation_fn=tf.nn.tanh, reuse=tf.AUTO_REUSE,
                                              scope=name + str(i), weights_initializer=layer.xavier_initializer(),
                                              trainable=trainable) for i in range(self.pars['n_freq'])]

        return tf.concat(logsigs, axis=1) if concat else logsigs

    def get_scaling_parameters(self, index):  # these should scale with gradient updates
        temp = tf.minimum((index + 1) / self.pars['temp_it'], 1)
        forget = tf.minimum((index + 1) / self.pars['forget_it'], 1)
        hebb_learn = tf.minimum((index + 1) / self.pars['hebb_learn_it'], 1)
        p2g_use = tf.sigmoid((index - self.pars['p2g_use_it']) / self.pars['p2g_scale'])  # from 0 to 1
        l_r = (self.pars['learning_rate_max'] - self.pars['learning_rate_min']) * (self.pars['l_r_decay_rate'] ** (
                index / self.pars['l_r_decay_steps'])) + self.pars['learning_rate_min']
        l_r = tf.maximum(l_r, self.pars['learning_rate_min'])
        g_cell_reg = 1 - tf.minimum((index + 1) / self.pars['g_reg_it'], 1)
        p_cell_reg = 1 - tf.minimum((index + 1) / self.pars['p_reg_it'], 1)
        ovc_cell_reg = 1 - tf.minimum((index + 1) / self.pars['ovc_reg_it'], 1)

        return temp, forget, hebb_learn, p2g_use, l_r, g_cell_reg, p_cell_reg, ovc_cell_reg

    def x2x_(self, x, x_):
        # Temporally filter sensorium (into 5 frequencies)
        x_s = [0] * self.pars['n_freq']
        for i in range(self.pars['n_freq']):
            with tf.variable_scope("x2x_" + str(i), reuse=tf.AUTO_REUSE):
                gamma = tf.get_variable("w_smooth_freq", [1], initializer=tf.constant_initializer(
                    np.log(self.pars['freq'][i] / (1 - self.pars['freq'][i]))),
                                        trainable=True)
            # Inverse sigmoid as initial parameter
            a = tf.cast(tf.sigmoid(gamma), tf.float64)
            # Filter
            x_s[i] = a * x_[i] + x * (1 - a)

        return x_s

    # PATH INTEGRATION
    def gen_g(self, g, d):
        # Generative prior on grids if first step in environment, else transition
        g, sigma = tf.cond(tf.constant(self.seq_pos > 0, dtype=tf.bool),
                           true_fn=lambda: self.g2g(g, d, self.pars['no_direc_gen'], name='gen'),
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
        update = tf.reshape(tf.matmul(t_mat, tf.reshape(g_p, [self.pars['batch_size'], self.pars['g_size'], 1])),
                            [self.pars['batch_size'], self.pars['g_size']])

        return update

    def g_prior(self, name=''):
        # Initial EC representation
        with tf.variable_scope("g_prior", reuse=tf.AUTO_REUSE):
            g = tf.tile(tf.get_variable("mu_g_prior" + name, [1, self.pars['g_size']],
                                        initializer=tf.truncated_normal_initializer(stddev=self.pars['g_init'])),
                        [self.pars['batch_size'], 1])
            logsig = tf.tile(tf.get_variable("logsig_g_prior" + name, [1, self.pars['g_size']],
                                             initializer=tf.truncated_normal_initializer(stddev=self.pars['g_init'])
                                             ), [self.pars['batch_size'], 1])

        sigma = tf.exp(logsig)

        return tf.cast(g, tf.float64), tf.cast(sigma, tf.float64)

    def get_transition(self, d, name=''):
        # Get transition matrix based on relationship / action
        d_mixed = fu_co(d, self.pars['d_mixed_size'], activation_fn=tf.tanh, reuse=tf.AUTO_REUSE,
                        scope='d_mixed_g2g' + name) if self.pars['d_mixed'] else d

        t_vec = tf.layers.dense(d_mixed, self.pars['g_size'] ** 2, activation=None, reuse=tf.AUTO_REUSE,
                                name='mu_g2g' + name, kernel_initializer=tf.zeros_initializer, use_bias=False)
        # Turn vector into matrix
        trans_all = tf.reshape(t_vec, [self.pars['batch_size'], self.pars['g_size'], self.pars['g_size']])
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

    # INFERENCE
    def inference(self, g2g_all, x, x_two_hot, x_):
        # Infer all variables
        x2p, x_s, _, x_comp = self.x2p(x, x_, x_two_hot)

        # Infer entorhinal representation
        g, p_x = self.infer_g(g2g_all, x2p, x)

        return

    def x2p(self, x, x_t, x_two_hot):
        # Provides input to place cell layer
        if self.pars['two_hot']:
            x_comp = x_two_hot
        else:
            x_comp = self.f_compress(x)

        # Temporally filter
        x_ = self.x2x_(x_comp, x_t)

        # Normalise
        x_normed = self.f_n(x_)

        # Tile to same size as hippocampus
        x_2p = self.x_2p(x_normed)

        return x_2p, x_, tf.concat(x_normed, axis=1), x_comp

    def x_2p(self, x_):
        ps = [0] * self.pars['n_freq']

        for i in range(self.pars['n_freq']):
            with tf.variable_scope("x_2p" + str(i), reuse=tf.AUTO_REUSE):
                w_p = tf.get_variable("w_p", [1], initializer=tf.constant_initializer(1.0))
            w_p = tf.cast(tf.sigmoid(w_p), tf.float64)

            # Tile to size of hippocampus
            ps[i] = tf.tile(w_p * x_[i], (1, self.pars['n_phases_all'][i]))

        p = tf.concat(ps, 1)

        return p

    def p2g(self, x2p, x):
        # Infer EC representation

        # Extract memory
        p_x = self.attractor(x2p, self.pars['which_way'][1])

        # Check if memory predicts data
        x_hat = x_hat_logits = self.f_x(p_x)

        return

    def infer_g(self, g2g_all, g_x2p, x):
        # Infers EC representation (grid cells)
        p_x = None
        g, sigma = g2g_all

        # Inference (factorise posteriors)
        g_p2g, sigma_p2g, p_x = self.p2g(g_x2p, x)

        return

    def f_n(self, x):
        x_normed = [0] * self.pars['n_freq']
        for i in range(self.pars['n_freq']):
            # Apply normalisation to each frequency
            with tf.variable_scope("f_n" + str(i), reuse=tf.AUTO_REUSE):
                x_normed[i] = x[i]
                # subtract mean and threshold
                x_normed[i] = tf.maximum(x_normed[i] - tf.reduce_mean(x_normed[i], axis=1, keepdims=True), 0)
                # l2 normalise
                x_normed[i] = tf.nn.l2_normalize(x_normed[i], axis=1)

        return x_normed

    def f_x(self, p):
        # Predicts sensory observation from HPC representation p
        ps = tf.split(value=p, num_or_size_splits=self.pars['n_place_all'], axis=1)

        # Same as W_tile^T
        x_s = tf.reduce_sum(tf.reshape(ps[self.pars['prediction_freq']], (self.pars['batch_size'],
                                                                          self.pars['n_phases_all'][
                                                                              self.pars['prediction_freq']],
                                                                          self.pars['s_size_comp'])), 1)

        with tf.variable_scope("f_x", reuse=tf.AUTO_REUSE):
            w_x = tf.get_variable("w_x", [1], initializer=tf.constant_initializer(1.0))
            b_x = tf.get_variable("bias", [self.pars['s_size_comp']], initializer=tf.constant_initializer(0.0))

        x_logits = w_x * x_s + b_x

        # Decompress sensory
        x_logits = self.f_decompress(x_logits)

        x = tf.nn.softmax(x_logits)

        return x, x_logits

    def f_compress(self, x):
        # Compress sensory data
        x_hidden = fu_co(x, self.pars['s_size_comp_hidden'], activation_fn=tf.nn.elu, reuse=tf.AUTO_REUSE,
                         scope='f_compress_1')
        x_comp = fu_co(x_hidden, self.pars['s_size_comp'], activation_fn=None, reuse=tf.AUTO_REUSE,
                       scope='f_compress_2')

        return x_comp

    def attractor(self, init, which_way):
        # Attractor network for retrieving memories
        if which_way == 'normal':
            hebb_diff_freq_its_max = self.pars['Hebb_diff_freq_its_max']
        else:
            hebb_diff_freq_its_max = self.pars['Hebb_inv_diff_freq_its_max']

        ps = [0] * (self.pars['n_recurs'] + 1)
        ps[0] = self.pars['p_activation'](init)

        for i in range(self.pars['n_recurs']):
            # Get Hebbian update
            update = self.heb_scal_prod(ps[i], i, which_way, hebb_diff_freq_its_max)

            ps_f = tf.split(value=ps[i], num_or_size_splits=self.pars['n_place_all'], axis=1)
            for f in range(self.pars['n_freq']):
                if i < hebb_diff_freq_its_max[f]:
                    ps_f[f] = self.pars['p_activation'](self.pars['prev_p_decay'] * ps_f[f] + update[f])
            ps[i + 1] = tf.concat(ps_f, axis=1)

        p = ps[-1]

        return p

    def heb_scal_prod(self, p, it_num, which_way, hebb_diff_freq_its_max):
        # Computation of scalar product updates (Ba et al. 2016)
        if which_way == 'normal':
            h_split = self.A_split
            a, b = self.mem_a_s, self.mem_b_s
            r_f_f = self.pars['R_f_F']
        else:
            h_split = self.A_inv_split
            a, b = self.mem_e_s, self.mem_f_s
            r_f_f = self.pars['R_f_F_inv']

        p_ = tf.transpose(tf.expand_dims(p, axis=2), [0, 2, 1])
        ps = tf.split(value=p, num_or_size_splits=self.pars['n_place_all'], axis=1)

        updates = [0] * self.pars['n_freq']
        updates_poss = self.hebb_scal_prod_helper(a, b, ps, r_f_f)

        for freq in range(self.pars['n_freq']):
            # More iterations for higher frequencies
            if it_num < hebb_diff_freq_its_max[freq] and np.sum(r_f_f, 0) > 0:
                hebb_add = tf.squeeze(tf.matmul(p_, h_split[freq]))
                updates[freq] = updates_poss[freq] + hebb_add

        return updates

    def hebb_scal_prod_helper(self, a, b, ps, r_f_f):
        # Computes scalar products of memories
        updates = [0] * self.pars['n_freq']
        scal_prods = []

        # Calculate scalar products for each frequency
        for freq in range(self.pars['n_freq']):
            p_freq = tf.expand_dims(ps[freq], 2)
            scal_prods.append(tf.matmul(tf.transpose(b[freq], [0, 2, 1]), p_freq))

        for freq in range(self.pars['n_freq']):  # Going downwards
            scal_prod = []
            for f in range(self.pars['n_freq']):  # Which frequencies influence which other frequencies
                if r_f_f[f][freq] > 0:
                    scal_prod.append(scal_prods[f])

            scal_prod_sum = tf.add_n(scal_prod)
            updates[freq] = tf.squeeze(tf.matmul(a[freq], scal_prod_sum))

        return updates

    # MEMORY FUNCTIONS
    def hebbian(self, p, p_g, p_x):
        a, b = p - p_g, p + p_g
        e, f = None, None
        if self.pars['hebb_type'] == [[2], [2]]:
            # Inverse
            e, f = p - p_x, p + p_x

        # add memories to a list
        self.mem_a, self.mem_a_s, self.mem_list_a, self.mem_list_a_s = self.mem_update(a, self.mem_list_a,
                                                                                       self.mem_list_a_s)
        self.mem_b, self.mem_b_s, self.mem_list_b, self.mem_list_b_s = self.mem_update(b, self.mem_list_b,
                                                                                       self.mem_list_b_s)
        if e is not None and f is not None:
            self.mem_e, self.mem_e_s, self.mem_list_e, self.mem_list_e_s = self.mem_update(e, self.mem_list_e,
                                                                                           self.mem_list_e_s)
            self.mem_f, self.mem_f_s, self.mem_list_f, self.mem_list_f_s = self.mem_update(f, self.mem_list_f,
                                                                                           self.mem_list_f_s)
        # 'forget' the Hebbian matrices
        self.A = self.A * self.forget * self.pars['lambd']
        self.A_split = tf.split(self.A, num_or_size_splits=self.par['n_place_all'], axis=2)

        self.A_inv = self.A_inv * self.forget * self.pars['lambd']
        self.A_inv_split = tf.split(self.A_inv, num_or_size_splits=self.pars['n_place_all'], axis=2)

        return


def direction_pars(pars_orig, pars, n_restart):
    n_envs = len(pars['widths'])
    b_s = int(pars['batch_size'])
    # Choose self.pars for current stage of training
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
