import parameters
import helper_functions
import copy as cp
import tensorflow as tf
import numpy as np
import importlib
import model as tem

pars = parameters.default_params()
pars_orig = pars.copy()

tf.reset_default_graph()
with tf.name_scope('Inputs'):
    it_num = tf.placeholder(tf.float32, shape=(), name='it_num')
    seq_ind = tf.placeholder(tf.float32, shape=(), name='seq_ind')
    rnn = tf.placeholder(tf.float32, shape=(pars['batch_size'], pars['p_size'], pars['p_size']), name='rnn')
    rnn_inv = tf.placeholder(tf.float32, shape=(pars['batch_size'], pars['p_size'], pars['p_size']), name='rnn')
    x1_two_hot = tf.placeholder(tf.float32, shape=(pars['batch_size'], pars['s_size_comp'], pars['seq_len']),
                                name='x1_two_hot')
    x1 = tf.placeholder(tf.float32, shape=(pars['batch_size'], pars['s_size'], pars['seq_len']), name='x')
    g_ = tf.placeholder(tf.float32, shape=(pars['batch_size'], pars['g_size']), name='g_')
    x_ = tf.placeholder(tf.float32, shape=(pars['batch_size'], pars['s_size_comp'] * pars['n_freq']), name='x_')
    sh = tf.placeholder(tf.float32, shape=(pars['batch_size'], pars['s_size']), name='shiny')
    # need to feed in lists etc
    d0 = tf.placeholder(tf.float32, shape=(pars['batch_size'], pars['n_actions'], pars['seq_len']), name='d')
    s_visi = tf.placeholder(tf.float32, shape=(pars['batch_size'], pars['seq_len']), name='s_visited')
    no_d = tf.placeholder(tf.float32, shape=(pars['batch_size']), name='no_direc_batch')
    x_two_hot = tf.unstack(x1_two_hot, axis=2)
    x = tf.unstack(x1, axis=2)
    d = tf.unstack(d0, axis=2)
    s_vis = tf.unstack(s_visi, axis=1)

# Initialise Model
model = tem.TEM(x, x_, x_two_hot, g_, d, rnn, rnn_inv, it_num, seq_ind, s_vis, sh, no_d, pars)

# Initialise Model Variables
fetches_all, fetches_summary, fetches_all_, fetches_summary_ = [], [], [], []
fetches_all.extend([model.g, model.p, model.p_g, model.x_gt, model.x_, model.A, model.A_inv, model.accuracy_gt,
                    model.train_op_all])
fetches_all_.extend([model.g, model.p, model.p_g, model.x_gt, model.x_, model.A, model.A_inv, model.accuracy_gt,
                     model.temp])
fetches_summary.extend([model.lx_p, model.lx_g, model.lx_gt, model.lp, model.lg, model.g, model.p, model.p_g,
                        model.x_gt, model.x_, model.A, model.A_inv, model.accuracy_p, model.accuracy_g,
                        model.accuracy_gt, model.merged, model.train_op_all])
fetches_summary_.extend([model.lx_p, model.lx_g, model.lx_gt, model.lp, model.lg, model.g, model.p, model.p_g,
                         model.x_gt, model.x_, model.A, model.A_inv, model.accuracy_p, model.accuracy_g,
                         model.accuracy_gt, model.merged, model.temp])

print('Graph Initialised')

"""Run Model"""
# Initialise Variables
lx_ps, lx_gs, lx_gts, lps, lgs = [], [], [], [], []
accs_p, accs_g, accs_gt = [], [], []
check_link_inference = False
acc_p, acc_g, acc_gt, seq_index, rn = 0, 0, 0, 0, 0
correct_link, positions_link, positions, visited, state_guess = None, None, None, None, None
position_all, direc_all = None, None
gs_all, ps_all, ps_gen_all, xs_all, gs_timeseries, ps_timeseries, pos_timeseries = \
    [None] * pars['n_envs_save'], [None] * pars['n_envs_save'], [None] * pars['n_envs_save'], \
    [None] * pars['n_envs_save'], [None] * pars['n_envs_save'], [None] * pars['n_envs_save'], \
    [None] * pars['n_envs_save']
cell_timeseries, prev_cell_timeseries = None, None
accs_x_to, accs_x_from = [None] * pars['n_envs_save'], [None] * pars['n_envs_save']
save_needed, save_ticker, summary_needed, summary_ticker, save_model = False, False, False, False, False
table, _ = helper_functions.combins_table(pars['s_size_comp'], 2)
n_restart = pars['restart_max'] + pars['curriculum_steps']
T, F, L, P = None, None, None, None
adjs, trans, states_mat, shiny_s, shiny_states = None, None, None, None, None
a_rnn, a_rnn_inv = None, None
gs, ps, x_s, x_data, start_state, prev_direc = None, None, None, None, None, None
n_walk = None
index = 0

print('Training Started')
for i in range(pars['train_iters']):
    # Curriculum of Behaviour Types
    pars, shiny_s, rn, n_restart, no_direc_batch = helper_functions.curriculum(pars_orig, pars, n_restart)

    # Make Environment
    adjs, trans, states_mat, shiny_states = helper_functions.make_environments(pars)

    # Create Hebbian Matrices
    a_rnn, a_rnn_inv = helper_functions.initialise_hebb(pars)

    # Initialise all other Variables
    gs, x_s, x_data, start_state, prev_direc, visited = helper_functions.initialise_variables(pars, adjs)

    # Collect Walking Data
    position_all, direc_all = helper_functions.get_walking_data(start_state, adjs, trans, prev_direc, shiny_states, n_walk * pars['n_walk'], pars)

    # Run Model
    for seq_index in range(n_walk):
        # Summary at End of Each Walk
        summary_needed = True if seq_index == n_walk - 1 and summary_ticker else False

        # Collect Data
        i1, i2 = seq_index * pars['n_walk'], (seq_index + 1) * pars['n_walk'] + 1
        walking_data = [position_all[:, i1:i2], direc_all[:, :, i1:i2 - 1]]
        new_data, old_data, model_vars = helper_functions.get_next_batch(walking_data, x_data, states_mat, index, visited, pars)
        xs, ds, position, visited, s_visited = new_data
        x_data, start_state = old_data
        T, F, L, P = model_vars
        xs_two_hot = helper_functions.onehot2twohot(xs, table, pars['s_size_comp'])

        feed_dict = {x1: xs, x_: x_s, x1_two_hot: xs_two_hot, g_: gs, d0: ds, rnn: a_rnn, rnn_inv: a_rnn_inv,
                     it_num: index, seq_ind: seq_index, s_visi: s_visited, sh: shiny_s, no_d: no_direc_batch}