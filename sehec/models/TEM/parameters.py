import numpy as np


def default_params():
    params = dict()

    params['batch_size'] = 16
    params['time_step_size'] = 1
    params['agent_step_size'] = 0.2
    params['discount'] = .9
    params['threshold'] = 1e-6
    params['lr_td'] = 1e-2
    params['t_episode'] = 25
    params['n_episode'] = 1
    params['state_density'] = 1
    params['twoDvalue'] = True

    params['room_width'] = 2
    params['room_depth'] = 2

    params['world_type'] = 'square'
    params['n_envs'] = params['batch_size']
    params['diff_env_batches_envs'] = np.arange(params['batch_size'])  # which batch in which environment

    params['widths'], params['n_states'], params['n_states_world'], params['n_actions'], params['jump_length'], \
        params['heights'] = get_states(params)

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
