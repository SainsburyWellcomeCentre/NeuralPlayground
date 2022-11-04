import numpy as np
import copy as cp
import tensorflow as tf


def default_params():
    params = dict()

    # Environment Parameters
    params['batch_size'] = 16
    params['world_type'] = 'square'
    params['diff_env_batches_envs'] = np.arange(params['batch_size'])
    params['state_density'] = 1
    params['n_envs'] = params['batch_size']
    params['diff_env_batches_envs'] = np.arange(params['batch_size'])  # which batch in which environment
    params['widths'], params['n_states'], params['n_states_world'], params['n_actions'], params['jump_length'], \
    params['heights'] = get_states(params)

    params['time_step_size'] = 1
    params['agent_step_size'] = 1
    params['discount'] = .9
    params['threshold'] = 1e-6
    params['t_episode'] = 25
    params['n_episode'] = 250
    params['train_iters'] = 300
    params['two_hot'] = True
    params['n_walk'] = params['t_episode']

    # Behaviour Parameters
    params['poss_behaviours'] = ['normal']  # '['shiny', 'normal'] for OVC cells, ['normal'] otherwise
    params['direc_bias'] = 0.2  # strength of that bias

    params['restart_max'] = np.ceil(6000 / params['t_episode']).astype(int)
    params['restart_min'] = np.ceil(2500 / params['t_episode']).astype(int)
    params['seq_jitter'] = np.ceil(200 / params['t_episode']).astype(int)
    params['curriculum_steps'] = 12 / params['t_episode']
    params['n_restart'] = params['restart_max'] + params['curriculum_steps']

    # Saving Parameters
    params['n_save_data'] = int(25 * max(params['n_states']) / params['t_episode'])
    params['n_envs_save'] = 6  # only save date from first X of batch
    params['save_interval'] = int(int(50000 / params['t_episode']) / params['n_save_data']) * params['n_save_data']
    params['save_model'] = 5 * params['save_interval']
    params['sum_int'] = 200
    params['link_inf_walk'] = int(3000 / params['t_episode'])

    # Model Parameters
    params['infer_g_type'] = 'g_p'
    params['two_hot'] = True

    n_phases_all = [10, 10, 8, 6, 6]  # numbers of variables for each frequency
    params['freq'] = [0.01, 0.7, 0.91, 0.97, 0.99, 0.9995]
    params['s_size'] = 45
    params['s_size_comp'] = 10
    params['s_size_comp_hidden'] = 20 * params['s_size_comp']

    params['n_phases_all'] = n_phases_all
    params['n_place_all'] = [p * params['s_size_comp'] for p in params['n_phases_all']]
    params['n_grids_all'] = [int(3 * n_phase) for n_phase in params['n_phases_all']]
    params['tot_phases'] = sum(params['n_phases_all'])
    params['stay_still'] = True
    params['p_size'] = int(params['tot_phases'] * params['s_size_comp'])
    params['g_size'] = sum(params['n_grids_all'])
    params['n_freq'] = len(params['n_phases_all'])
    params['prediction_freq'] = 1

    # Training Parameters
    params['no_direc_gen'] = False
    params['no_direction'] = None
    params['train_on_visited_states_only'] = True
    params['learning_rate_max'] = 9.4e-4
    params['learning_rate_min'] = 8e-5
    params['train_sig_g2g'] = True if 'g' in params['infer_g_type'] else False
    params['train_sig_p2g'] = True if 'p' in params['infer_g_type'] else False
    params['logsig_offset'] = -2
    params['logsig_ratio'] = 6

    # losses
    params['which_costs'] = ['lx_p', 'lx_g', 'lx_gt', 'lp', 'lg', 'lg_reg', 'lp_reg']
    if 'p' in params['infer_g_type']:
        params['which_costs'].append('lp_x')

    # regularisation values
    params['g_reg_pen'] = 0.01
    params['p_reg_pen'] = 0.02
    params['weight_reg_val'] = 0.001

    # Activations
    params['p_activation'] = lambda x: tf.nn.leaky_relu(tf.minimum(tf.maximum(x, -1), 1), alpha=0.01)
    params['g2g_activation'] = lambda x: tf.minimum(tf.maximum(x, -1), 1)

    # Initialisations
    params['g_init'] = 0.5
    params['p2g_init'] = 0.1

    # Number gradient updates for annealing
    params['temp_it'] = 2000
    params['forget_it'] = 200
    params['hebb_learn_it'] = 16000
    params['p2g_use_it'] = 400
    params['p2g_scale'] = 200
    params['p2g_sig_val'] = 10000
    params['ovc_reg_it'] = 4000
    params['g_reg_it'] = 40000000
    params['p_reg_it'] = 4000
    params['l_r_decay_steps'] = 4000
    params['l_r_decay_rate'] = 0.5

    # Hebbian
    params['hebb_mat_max'] = 1
    params['lambd'] = 0.9999
    params['eta'] = 0.5
    params['hebb_type'] = [[2], [2]]

    # Types of allowed connections in Hebbian matrices
    separate = [[1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]]

    hierarchical = [[1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 1]]

    hierarchical_t = [[1, 0, 0, 0, 0, 0],
                      [1, 1, 0, 0, 0, 0],
                      [1, 1, 1, 0, 0, 0],
                      [1, 1, 1, 1, 0, 0],
                      [1, 1, 1, 1, 1, 0],
                      [1, 1, 1, 1, 1, 1]]

    all2all = [[1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1]]

    # STATE TRANSITION [R_G_F_f says how frequency f influences frequency F (opposite to R_F_f_F)]
    params['R_f_F'] = cp.deepcopy(hierarchical_t)
    params['R_f_F_inv'] = cp.deepcopy(all2all)
    params['R_G_F_f'] = cp.deepcopy(hierarchical)

    params['mask_p'] = place_mask(params['n_phases_all'], params['s_size_comp'], params['R_f_F'])
    params['mask_g'] = grid_mask(params['n_grids_all'], params['R_G_F_f'])
    params['d_mixed'] = True
    params['d_mixed_size'] = 15 if params['world_type'] == 'square' else 20

    # PLACE ATTRACTOR
    params['n_recurs'] = params['n_freq']
    params['prev_p_decay'] = 0.8
    params['which_way'] = ['normal', 'normal'] if len(params['hebb_type']) < 2 else ['normal', 'inv']
    params['Hebb_diff_freq_its_max'] = [params['n_recurs'] - freq for freq in range(params['n_recurs'])]
    params['Hebb_inv_diff_freq_its_max'] = [params['n_recurs'] for _ in range(params['n_recurs'])]

    return params


def get_states(pars):
    world_type, n_envs = pars['world_type'], pars['n_envs']
    poss_heights = [8, 8, 9, 9, 11, 11, 12, 12, 8, 8, 9, 9, 11, 11, 12, 12]

    if world_type == 'square':
        poss_widths = [10, 10, 11, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 8, 9, 9]
        n_states = [x ** 2 for x in poss_widths]
        n_actions = 4

    else:
        n_states = None
        n_actions = None
        poss_widths = None
        poss_heights = None

    poss_widths = list(np.tile(poss_widths, int(np.ceil(n_envs / len(poss_widths))))[:n_envs])
    poss_heights = list(np.tile(poss_heights, int(np.ceil(n_envs / len(poss_heights))))[:n_envs])

    if world_type == 'square':
        n_states_world = [x ** 2 for x in poss_widths]

    else:
        n_states_world = None

    jump_length = [x - 2 for x in poss_widths]

    return poss_widths, n_states, n_states_world, n_actions, jump_length, poss_heights


def grid_mask(n_phases, r):
    g_size = sum(n_phases)
    cum_phases = np.cumsum(n_phases)

    c_p = np.insert(cum_phases, 0, 0).astype(int)

    mask = np.zeros((g_size, g_size), dtype=np.float32)

    for freq_row in range(len(n_phases)):
        for freq_col in range(len(n_phases)):
            mask[c_p[freq_row]:c_p[freq_row + 1], c_p[freq_col]:c_p[freq_col + 1]] = r[freq_row][freq_col]

    return mask


def place_mask(n_phases, s_size, rff):
    # mask - only allow across freq, within sense connections
    # p_size : total place cell size
    # s_size : total number of senses
    # n_phases : number of phases in each frequency
    tot_phases = sum(n_phases)
    p_size = s_size * tot_phases
    cum_phases = np.cumsum(n_phases)

    c_p = np.insert(cum_phases * s_size, 0, 0).astype(int)

    mask = np.zeros((p_size, p_size), dtype=np.float32)

    for freq_row in range(len(n_phases)):
        for freq_col in range(len(n_phases)):
            mask[c_p[freq_row]:c_p[freq_row + 1], c_p[freq_col]:c_p[freq_col + 1]] = rff[freq_row][freq_col]

    return mask
