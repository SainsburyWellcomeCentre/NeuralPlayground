import tensorflow as tf
from parameters import *
from environments import *
import tem
import warnings
warnings.filterwarnings("ignore")

save_path = r"C:\Users\Coursework\Documents\MSc Machine Learning\Project\TEM2\save"
"""MODEL"""
pars = default_params()
pars_orig = pars.copy()

tf.reset_default_graph()
print('Initialising Graph...')

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
    ## need to feed in lists etc
    d0 = tf.placeholder(tf.float32, shape=(pars['batch_size'], pars['n_actions'], pars['seq_len']), name='d')
    s_visi = tf.placeholder(tf.float32, shape=(pars['batch_size'], pars['seq_len']), name='s_visited')
    no_d = tf.placeholder(tf.float32, shape=(pars['batch_size']), name='no_direc_batch')
    x_two_hot = tf.unstack(x1_two_hot, axis=2)
    x = tf.unstack(x1, axis=2)
    d = tf.unstack(d0, axis=2)
    s_vis = tf.unstack(s_visi, axis=1)

model = tem.TEM(x, x_, x_two_hot, g_, d, rnn, rnn_inv, it_num, seq_ind, s_vis, sh, no_d, pars)

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
print('Graph Initialised.')

# CREATE SESSION
sess = tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=1)  # saves variables learned during training
tf.global_variables_initializer().run()
tf.get_default_graph().finalize()

""" RUN MODEL """
# INITIALISE VARIABLES
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
table, _ = combins_table(pars['s_size_comp'], 2)
n_restart = pars['restart_max'] + pars['curriculum_steps']
T, F, L, P = None, None, None, None
adjs, trans, states_mat, shiny_s, shiny_states = None, None, None, None, None
a_rnn, a_rnn_inv = None, None
gs, ps, x_s, x_data, start_state, prev_direc = None, None, None, None, None, None
n_walk = None
index = 0
## width of env for particular batch is pars['widths'][pars['diff_env_batches_envs'][env]]

print('Training Started') if pars['training'] else print('Debugging Started')
for i in range(pars['train_iters']):

    # INITIALISE ENVIRONMENT AND INPUT VARIABLES
    msg = 'New Environment ' + str(i) + ' ' + str(index) + ' ' + str(index * pars['seq_len'])
    print(msg)

    # curriculum of behaviour types
    pars, shiny_s, rn, n_restart, no_direc_batch = curriculum(pars_orig, pars, n_restart)

    if save_ticker:
        save_needed = True
        gs_all, ps_all, ps_gen_all, xs_all, gs_timeseries, ps_timeseries, pos_timeseries = \
            [None] * pars['n_envs_save'], [None] * pars['n_envs_save'], [None] * pars['n_envs_save'], \
            [None] * pars['n_envs_save'], [None] * pars['n_envs_save'], [None] * pars['n_envs_save'], \
            [None] * pars['n_envs_save']
        accs_x_to, accs_x_from = [None] * pars['n_envs_save'], [None] * pars['n_envs_save']
        n_walk = pars['n_save_data']
    elif i % 10 in [5]:  # check for link inference every 10 environments
        check_link_inference, positions_link, correct_link, state_guess = True, [None] * pars['batch_size'], \
                                                                          [None] * pars['batch_size'], [None] * pars[
                                                                              'batch_size']
        n_walk = pars['link_inf_walk']
    else:
        n_walk = int(n_restart) + rn

    # make environemnts
    adjs, trans, states_mat, shiny_states = make_environments(pars)

    # initialise Hebbian matrices
    a_rnn, a_rnn_inv = initialise_hebb(pars)

    # initialise all other variables
    gs, x_s, x_data, start_state, prev_direc, visited = initialise_variables(pars, adjs)

    # Collect full sequence of data
    position_all, direc_all = get_walking_data(start_state, adjs, trans, prev_direc, shiny_states,
                                               n_walk * pars['n_walk'], pars)

    # run model
    for seq_index in range(n_walk):
        # summary at end of each walk
        summary_needed = True if seq_index == n_walk - 1 and summary_ticker else False

        # COLLECT DATA
        i1, i2 = seq_index * pars['n_walk'], (seq_index + 1) * pars['n_walk'] + 1
        walking_data = [position_all[:, i1:i2], direc_all[:, :, i1:i2 - 1]]
        new_data, old_data, model_vars = get_next_batch(walking_data, x_data, states_mat, index, visited, pars)
        xs, ds, position, visited, s_visited = new_data
        x_data, start_state = old_data
        T, F, L, P = model_vars
        xs_two_hot = onehot2twohot(xs, table, pars['s_size_comp'])

        feed_dict = {x1: xs, x_: x_s, x1_two_hot: xs_two_hot, g_: gs, d0: ds, rnn: a_rnn, rnn_inv: a_rnn_inv,
                     it_num: index, seq_ind: seq_index, s_visi: s_visited, sh: shiny_s, no_d: no_direc_batch}

        fetch = fetches_all_ if check_link_inference or save_needed else fetches_all
        results = sess.run(fetch, feed_dict)
        gs, ps, ps_gen, x_gt, x_s, a_rnn, a_rnn_inv, acc_gt, _ = results

        # Check for nans etc.
        for ar, array in enumerate([gs, ps, ps_gen, x_gt, x_s]):
            if np.isnan(array).any():
                raise ValueError('Nan in array ' + str(ar))

sess.close()
print('Finished training')

# SAVE DATA
markers = [lx_ps, lx_gs, lx_gts, lps, lgs, accs_p, accs_g, accs_gt]
np.save(save_path + '/markers', markers)