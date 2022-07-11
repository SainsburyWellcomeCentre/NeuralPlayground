import sys

sys.path.append("../")
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.layers as layer
import scipy
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
        self.gamma = mod_kwargs["discount"]
        self.threshold = mod_kwargs["threshold"]
        self.learning_rate = mod_kwargs["lr_td"]
        self.t_episode = mod_kwargs["t_episode"]
        self.room_width = mod_kwargs["room_width"]
        self.room_depth = mod_kwargs["room_depth"]
        self.state_density = mod_kwargs["state_density"]
        twoD = mod_kwargs['twoD']
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

        self.objects = np.zeros(shape=(self.batch_size, self.s_size, self.t_episode))
        for i in range(self.batch_size):
            for j in range(self.t_episode):
                rand = random.randrange(0, self.s_size)
                self.objects[i][rand][j] = 1
                # random.shuffle(self.objects[:][0])

        # Variables for the SR-agent state spaceg
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
        if twoD == True:
            adjs, trans = square_world(self.w)
            # self.create_transmat(self.state_density,  '2D_env')

        # Initialise Variables for TEM
        self.a_rnn, self.a_rnn_inv = initialise_hebbian(self)
        self.gs, self.x_s, self.visited = initialise_variables(self)

        # Print Testing
        print("Transition matrix: ")
        print(trans)
        print("n_states: ", self.n_state)
        print("First sensory observation: ", self.objects[0, :, 0])
        print("Size of g: ", self.g_size)

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
        curr_state = index
        x = self.objects[:, :, index]

        return curr_state, x

    def act(self, obs):
        actions = np.zeros((self.batch_size, 2, self.t_episode))
        xs = np.zeros((self.batch_size, self.s_size, self.t_episode))
        for i in range(self.t_episode):
            self.obs_history.append(obs)
            if len(self.obs_history) >= 1000:
                self.obs_history = [obs, ]

            arrow = [[0, 1], [0, -1], [1, 0], [-1, 0]]
            action = np.random.normal(scale=0.1, size=(2,))
            diff = action - arrow
            dist = np.sum(diff ** 2, axis=1)
            index = np.argmin(dist)
            action = arrow[index]
            next_state, next_object = self.obs_to_state(obs)

            actions.append(action)
            xs.append(next_object[0])
        print(xs)
        x_, x_two_hot = calculate(self, xs)

        return actions, xs, x_, x_two_hot


def calculate(self, x):
    i = len(self.obs_history)
    # Initialisation of filtered x
    if len(self.obs_history) == 1:
        x_t = self.x_s
        g_t = self.gs
    # Pointing to previous x
    else:
        x_t = self.x_[i - 1]
        g_t = self.g[i - 1]

    # Two-hot Encoding
    x_two_hot = onehot2twohot(self, x, self.table, self.s_size_comp)

    # Temporally filter
    x_ = x2x_(self, self.x_two_hot, x_t)

    # # Generative Transition
    # g_gen, g2g_all = gen_g(self, g_t, action)

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
    " Produces a table of s_size two-hot encoded vectors, each of size 10."
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

    return table, rev_table


def onehot2twohot(self, onehot, table, compress_size):
    seq_len = np.shape(onehot)[2]
    batch_size = np.shape(onehot)[0]
    twohot = np.zeros((batch_size, compress_size, seq_len))
    for i in range(seq_len):
        vals = np.argmax(onehot[:, :, i], 1)
        for b in range(batch_size):
            twohot = table[vals][int(b)]

    return twohot


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
    mu, sigma = tf.cond(self.seq_pos > 0, true_fn=lambda: self.g2g(g, d, self.no_direc_gen, name='gen'),
                        false_fn=lambda: self.g_prior())

    # the same but for used for inference network
    mu_inf, sigma_inf = tf.cond(self.seq_pos > 0, true_fn=lambda: self.g2g(g, d, False, name='inf'),
                                false_fn=lambda: self.g_prior())

    return mu, (mu_inf, sigma_inf)


def g2g(self, g, d, no_direc=False, name=''):
    """make grid to grid transisiton"""
    # transition update
    update = self.get_g2g_update(g, d, no_direc, name='')
    # add on update to current representation
    mu = update + g
    # apply activation
    mu = self.f_g(mu)
    # get variance
    logsig = self.hierarchical_logsig(g, 'sig_g2g' + name, self.n_grids_all, self.n_grids_all,
                                      self.par['train_sig_g2g'], concat=True)
    logsig += self.par['logsig_offset']

    sigma = tf.exp(logsig)

    return mu, sigma


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
        logsig = tf.tile(tf.get_variable("logsig_g_prior" + name, [1, self.par['g_size']],
                                         initializer=tf.truncated_normal_initializer(stddev=self.g_init)),
                         [self.par['batch_size'], 1])

    sigma = tf.exp(logsig)

    return mu, sigma


# ------------------------------------------------------------------------------------------------------
# MAIN
env_name = "env_example"
pars = default_params()
# Initialise Environment(s)
env = TEMenv(environment_name=env_name, room_width=pars['room_width'], room_depth=pars['room_depth'],
             time_step_size=pars['time_step_size'], agent_step_size=pars['agent_step_size'],
             stay_still=pars['stay_still'], p_size=pars['p_size'], g_size=pars['g_size'], g_init=pars['g_init'],
             s_size_comp=pars['s_size_comp'], n_freq=pars['n_freq'], n_state=pars['n_state'])

agent = TEM(discount=pars['discount'], t_episode=pars['t_episode'], threshold=pars['threshold'], lr_td=pars['lr_td'],
            room_width=pars['room_width'], room_depth=pars['room_depth'], state_density=pars['state_density'],
            twoD=pars['twoDvalue'])

total_iters = 0
obs, state = env.reset()
obs = obs[:2]
xs = []
# actions = [[-1,0], [0,1], [-1,0], [0,1], [-1,0]]
for i in range(pars['n_episode']):
    # Initialise Environment, Weight and Variable Batch
    adjs, trans = [], []
    for width in pars['widths']:
        adj, tran = env.square_world(width, pars['stay_still'])
        adjs.append(env)
        trans.append(tran)
    a_rnn, a_rnn_inv = env.initialise_hebbian()
    gs, x_s, visited = env.initialise_variables()

    # action = actions[j]
    for j in range(pars['t_episode']):
        actions, x, x_, x_two_hot = agent.act(obs)
        obs, state, reward = env.step(actions[j])
        obs = obs[:2]
    xs.append(xs)
    total_iters += 1

# print(np.shape(x), x_, np.shape(x_two_hot))
env.plot_trajectory()
