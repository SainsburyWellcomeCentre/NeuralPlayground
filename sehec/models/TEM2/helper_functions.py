from environments import *
import numpy as np
import tensorflow as tf
import copy as cp
import os
import datetime
import logging
from distutils.dir_util import copy_tree
import scipy
import inspect

def place_mask(n_phases, s_size, rff):
    # mask - only allow across freq, within sense connections
    # p_size : total place cell size
    # s_size : total number of senses
    # n_phases : number of phases in each frequency
    tot_phases = sum(n_phases)
    p_size = s_size * tot_phases
    cum_phases = np.cumsum(n_phases)

    c_p = np.insert(cum_phases*s_size, 0, 0).astype(int)

    mask = np.zeros((p_size, p_size), dtype=np.float32)

    for freq_row in range(len(n_phases)):
        for freq_col in range(len(n_phases)):
            mask[c_p[freq_row]:c_p[freq_row+1], c_p[freq_col]:c_p[freq_col+1]] = rff[freq_row][freq_col]

    return mask


def grid_mask(n_phases, r):
    g_size = sum(n_phases)
    cum_phases = np.cumsum(n_phases)

    c_p = np.insert(cum_phases, 0, 0).astype(int)

    mask = np.zeros((g_size, g_size), dtype=np.float32)

    for freq_row in range(len(n_phases)):
        for freq_col in range(len(n_phases)):
            mask[c_p[freq_row]:c_p[freq_row + 1], c_p[freq_col]:c_p[freq_col + 1]] = r[freq_row][freq_col]

    return mask

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
        # forward mapping
        c = combins(n, k, m)
        if map_max is None or m < map_max:
            table.append(c)
            rev_table[c] = m
        else:
            rev_table[c] = m % map_max
    return table, rev_table

def initialise_hebb(par):
    a_rnn = np.zeros((par['batch_size'], par['p_size'], par['p_size']))
    a_rnn_inv = np.zeros((par['batch_size'], par['p_size'], par['p_size']))
    return a_rnn, a_rnn_inv

def initialise_variables(par, adjs):
    gs = np.maximum(np.random.randn(par['batch_size'], par['g_size']) * par['g_init'], 0)
    x_s = np.zeros((par['batch_size'], par['s_size_comp'] * par['n_freq']))
    x_data = np.zeros((par['batch_size'], par['s_size'], par['n_walk'] + 1))

    n_states = par['n_states_world']
    visited = np.zeros((par['batch_size'], max(n_states)))

    envs = par['diff_env_batches_envs']

    start_state = np.zeros(par['batch_size'])
    for batch in range(par['batch_size']):
        # this needs to be sorted out for hex not box worlds
        allowed_states = np.where(np.sum(adjs[envs[batch]], 1) > 0)[0]  # only include states you can get to
        if par['world_type'] in ['loop_laps']:
            start_state[batch] = 0
        else:
            start_state[batch] = np.random.choice(allowed_states)
    prev_direc = np.random.randint(0, par['n_actions']+1, par['batch_size']).astype(int)

    return gs, x_s, x_data, start_state, prev_direc, visited

def get_next_batch(walking_data, x_data, states_mat, index, visited, pars):
    envs = pars['diff_env_batches_envs']

    position, direc = walking_data

    data = get_new_data_diff_envs(position, x_data, envs, states_mat, pars)

    xs = data[:, :, 1: pars['seq_len'] + 1]
    ds = direc[:, :, :pars['seq_len']]

    x_data, start_state = data, position[:, -1].astype(int)

    # position is n_seq + 1, where 1st element is 'start state position'
    if pars['train_on_visited_states_only']:
        s_visited = np.zeros((pars['batch_size'], pars['seq_len']))
        for b in range(pars['batch_size']):
            for seq in range(pars['seq_len']):
                pos = int(position[b, seq + 1])  # 'current' position (+1 as that's where we start)
                s_visited[b, seq] = 1 if visited[b, pos] == 1 else 0  # have I visited this position before
                visited[b, pos] = 1  # add position to places I've been
    else:
        s_visited = np.ones((pars['batch_size'], pars['seq_len']))

    new_data = (xs, ds, position, visited, s_visited)
    old_data = (x_data, start_state)

    temp = np.minimum(1, (index + 1) / pars['temp_it'])
    forget = np.minimum(1, (index + 1) / pars['forget_it'])
    hebb_learn = np.minimum(1, (index + 1) / pars['hebb_learn_it'])
    p2g_use = np.minimum(1, (index + 1) / pars['p2g_use_it'])

    model_vars = [temp, forget, hebb_learn, p2g_use]

    return new_data, old_data, model_vars

def onehot2twohot(onehot, table, compress_size):
    seq_len = np.shape(onehot)[2]
    batch_size = np.shape(onehot)[0]
    twohot = np.zeros((batch_size, compress_size, seq_len))
    for i in range(np.shape(onehot)[2]):
        vals = np.argmax(onehot[:, :, i], 1)
        for b in range(np.shape(onehot)[0]):
            twohot[b, :, i] = table[vals[int(b)]]

    return twohot

def sparse_softmax_cross_entropy_with_logits(labels, logits):
    labels = tf.argmax(labels, 1)
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

def squared_error(t, o, keep_dims=False):
    return 0.5 * tf.reduce_sum(tf.square(t - o), 1, keepdims=keep_dims)

def acc_tf(real, pred):
    correct_prediction = tf.equal(tf.argmax(real, 1), tf.argmax(pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return tf.cast(accuracy * 100, tf.int32)

def combine2(mu1, mu2, sigma1, sigma2, batch_size):
    out_size = tf.shape(mu1)[1]
    inv_sigma_sq1 = tf.truediv(1.0, tf.square(sigma1))
    inv_sigma_sq2 = tf.truediv(1.0, tf.square(sigma2))

    logsigma = -0.5 * tf.log(inv_sigma_sq1 + inv_sigma_sq2)
    sigma = tf.exp(logsigma)

    mu = tf.square(sigma) * (mu1 * inv_sigma_sq1 + mu2 * inv_sigma_sq2)
    e = tf.random_normal((batch_size, out_size), mean=0, stddev=1)
    return mu + sigma * e, mu, logsigma, sigma

def tf_repeat_axis_1(tensor, repeat, dim1):
    dim0 = tf.shape(tensor)[0]
    return tf.reshape(tf.tile(tf.reshape(tensor, (-1, 1)), (1, repeat)), (dim0, dim1))
