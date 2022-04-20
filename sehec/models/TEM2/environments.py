import numpy as np
import copy as cp

def make_environments(par):
    env = 0
    n_envs = len(par['widths'])
    states_mat = [0] * n_envs
    shiny_states = [0] * n_envs
    n_senses = [par['s_size']] * par['n_freq']

    width = par['widths'][env]
    height = par['heights'][env]
    adj, tran = square_world(width, par['stay_still'])

    states_mat[env], shiny_states[env] = torus_state_data(n_senses, adj, env, par)


def square_world(width, stay_still):
    """
        #state number counts accross then down
        a = np.asarray(range(25))
        print(a)
        print(np.reshape(a,(5,5)))
        [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]
         [20 21 22 23 24]]
    """
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

    return adj, tran

def torus_state_data(n_senses, adj, env, par):
    width = par['widths'][env]
    shiny_bias = par['shiny_bias_env'][env]
    shiny_sense = par['shiny_sense'][env]
    n_states = par['n_states_world'][env]

    n_freq = np.size(n_senses)
    states_vec = np.zeros((n_states, 1))
    shiny_use = True if shiny_bias[0] > 0 else False
    choices = np.arange(n_senses[0])
    shiny_states = None

    if shiny_use:
        max_sep = np.maximum((width - 2) / len(shiny_sense), 4)

        shiny_states = []
        while len(shiny_states) < 2:
            shiny_states = []
            # choose shiny state so not on boundary
            allowed_states = [x for x in range(n_states) if np.sum(adj, 0)[x] == np.max(np.sum(adj, 0))]

            for i in range(len(shiny_sense)):
                # choose shiny position
                s_s = np.random.choice(allowed_states)
                shiny_states.append(s_s)
                # update allowed states given shiny position
                if i < len(shiny_sense) - 1:
                    allowed_states = [x for x in allowed_states if
                                      np.min(distance_between_states(x, s_s, width, par['world_type'])) > max_sep]

                if not allowed_states:
                    print('No space to put object ' + str(i + 2), shiny_states)
                    break
            max_sep += -0.5  # reduce max_sep if cant find space to put at least 2 shinies in each room
        print(max_sep + 0.5, len(shiny_states))

    if shiny_use:
        # remove shiny senses from available sense
        shiny_sense_sorted = sorted(list(set(shiny_sense)), reverse=True)
        for s_s in shiny_sense_sorted:
            # this requires choices be ordered + sense not repeated (hence set)
            choices = np.delete(choices, s_s)

    if par['world_type'] in ['loop_laps']:
        # choose reward sense
        reward_sense = np.random.choice(choices)
        # choices = np.delete(choices, reward_sense)
    else:
        reward_sense = 0

    for i in range(n_states):
        if par['world_type'] == 'loop_laps':
            new_state = np.random.choice(choices)
            len_loop = int(n_states / par['n_laps'])

            states_vec[i, 0] = new_state if i / len_loop < 1 else states_vec[i - len_loop, 0]

        else:
            # choose which sense goes where
            new_state = np.random.choice(choices)
            states_vec[i, 0] = new_state

    if par['world_type'] in ['loop_laps']:
        # make particular position special in track
        states_vec[par['reward_pos'], 0] = reward_sense

    if shiny_use:
        # put shinies in state_mat
        for sense, state in zip(shiny_sense, shiny_states):
            # assign sense to states
            states_vec[state, 0] = sense

    states_mat = np.repeat(states_vec, n_freq, axis=1)
    return states_mat, shiny_states

def walk_square(adj, tran, time_steps, start_state, prev_dir, shiny_state, env, params):
    """
        #state number counts accross then down
        a = np.asarray(range(25))
        print(a)
        print(np.reshape(a,(5,5)))
        [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]
         [20 21 22 23 24]]
    """
    position = np.zeros((1, time_steps + 1), dtype=np.int16)
    direc = np.zeros((4, time_steps))
    if params['world_type'] == 'rectangle':
        width = params['widths'][env]
    else:
        width = int(np.sqrt(np.shape(adj)[0]))

    shiny_bias = params['shiny_bias_env'][env]

    sb_min = shiny_bias[0]
    sb_max = shiny_bias[1]

    shiny_b, rn, shiny_s_ind, shiny_recent, ps_current = None, None, None, None, None
    current_angle = np.random.uniform(-np.pi, np.pi)

    # consider rectangular environment. if move +1=up, -1=down, +length=right, -length=left.
    if params['world_type'] == 'rectangle':
        height, wid = params['heights'][env], params['widths'][env]
        if height * wid != len(adj):
            raise ValueError('incorrect heigh/width : height * width not equal to number of states')
    else:
        height, wid = width, width

    distance_index = 1  # euclidean=0 , steps=1
    if sb_min > 0:
        # object want to go to
        shiny_s_ind = np.random.choice(np.arange(len(shiny_state)))
        shiny_b = [shiny_s_ind, shiny_state[shiny_s_ind], 0]
        rn = np.random.randint(params['object_hang_min'], params['object_hang_max'])

    position[0, 0] = int(start_state)
    for i in range(time_steps):
        available = np.where(tran[int(position[0, i]), :] > 0)[0].astype(int)

        # head towards objects, or in straight lines
        if sb_min > 0:
            # bias towards objects
            # choose new object to go to
            if shiny_b[2] > rn:
                try:
                    shiny_s_ind = np.random.choice([x for x in range(len(shiny_state)) if x != shiny_s_ind])
                except ValueError:
                    shiny_s_ind = np.random.choice([x for x in range(len(shiny_state))])
                shiny_b = [shiny_s_ind, shiny_state[shiny_s_ind], 0]
                rn = np.random.randint(params['object_hang_min'], params['object_hang_max'])
            # visited current shiny
            if position[0, i] == shiny_b[1]:
                shiny_b[2] += 1

            distances = [distance_between_states(shiny_state[shiny_s_ind], x, width, params['world_type'])
                         [distance_index] for x in available]

            ps = [1 / x for x in distances]
            ps = [x / sum(ps) for x in ps]

            # when in vicinity of object, move there more voraciously.
            # i.e. when not in vicinity this allows better exploration

            # bias to current object of choice
            g = np.zeros_like(available).astype(np.float32)
            min_dis_ind = np.random.choice(np.where(distances == min(distances))[0])
            g[min_dis_ind] = 1

            p = (sb_min * g) + (1 - sb_min - sb_max) * tran[int(position[0, i]), available] + sb_max * np.asarray(ps)

            # Staying still should always occur a certain proportion of time of time
            stay_still_pos = np.where(available == int(position[0, i]))[0]
            if len(stay_still_pos) > 0:
                p = (1 - params['object_stay_still']) * p / sum(p[np.arange(len(p)) != stay_still_pos[0]])
                p[stay_still_pos[0]] = params['object_stay_still']
            new_poss_pos = np.random.choice(available, p=p)
        elif params['bias_type'] == 'angle':
            new_poss_pos, current_angle = move_straight_bias(current_angle, position[0, i], width, available, tran,
                                                             params)
        else:
            new_poss_pos = np.random.choice(available)

        if adj[position[0, i], new_poss_pos] == 1:
            position[0, i + 1] = new_poss_pos
        else:
            position[0, i + 1] = int(cp.deepcopy(position[0, i]))

        prev_dir, _ = rectangle_relation(position[0, i], position[0, i + 1], wid, height)
        if prev_dir < 4:
            direc[prev_dir, i] = 1
        # stay still is just a set of zeros

    return position, direc, prev_dir

def move_straight_bias(current_angle, position, width, available, tran, params):
    # angle is allo-centric
    # from available position - find distance and angle from current pos
    angle_checker = angle_between_states_square
    diff_angle_min = np.pi / 4

    angles = [angle_checker(position, x, width) if x != position else 10000 for x in available]
    # find angle closest to current angle
    a_diffs = [np.abs(a - current_angle) for a in angles]
    a_diffs = [a if a < np.pi else np.abs(2 * np.pi - a) for a in a_diffs]

    angle_diff = np.min(a_diffs)

    if angle_diff < diff_angle_min:
        a_min_index = np.where(a_diffs == angle_diff)[0][0]
        angle = current_angle
    else:  # hit a wall - then do random non stationary choice
        p_angles = [1 if a < 100 else 0.000001 for a in angles]
        a_min_index = np.random.choice(np.arange(len(available)), p=np.asarray(p_angles) / sum(p_angles))
        angle = angles[a_min_index]

    new_poss_pos = int(available[a_min_index])

    angle += np.random.uniform(-params['angle_bias_change'], params['angle_bias_change'])
    angle = np.mod(angle + np.pi, 2 * np.pi) - np.pi  # keep between +- pi

    if np.random.rand() > params['direc_bias']:
        p = tran[int(position), available]
        new_poss_pos = np.random.choice(available, p=p)

    return new_poss_pos, angle

def angle_between_states_square(s1, s2, width):
    x1 = s1 % width
    x2 = s2 % width

    y1 = np.floor(s1 / width)
    y2 = np.floor(s2 / width)

    angle = np.arctan2(y1 - y2, x2 - x1)

    return angle

def distance_between_states(s1, s2, width, world_type):
    x1 = s1 % width
    x2 = s2 % width

    y1 = np.floor(s1 / width)
    y2 = np.floor(s2 / width)

    if world_type == 'hex':
        level_1 = np.mod(y1, 2)
        level_2 = np.mod(y2, 2)

        x1 += -level_1 * 0.5
        x2 += -level_2 * 0.5
        y1 *= np.sqrt(3) / 2
        y2 *= np.sqrt(3) / 2

    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) + 1e-6
    steps = np.abs(x1 - x2) + np.abs(y1 - y2) + 1e-6
    return distance, steps

def rectangle_relation(s1, s2, width, height):
    # consider square environment. if move +1=up, -1=down, +length=right, -length=left.
    diff = s2 - s1
    if diff == width or diff == -width * (height - 1):  # down
        direc = 0
        name = 'down'
    elif diff == -width or diff == width * (height - 1):  # up
        direc = 1
        name = 'up'
    elif diff == -1 or diff == (width - 1):  # left
        direc = 2
        name = 'left'
    elif diff == 1 or diff == -(width - 1):  # right
        direc = 3
        name = 'right'
    elif diff == 0:
        direc = 4
        name = 'stay still'
    else:
        raise ValueError('impossible action')

    return direc, name

def curriculum(pars_orig, pars, n_restart):
    n_envs = len(pars['widths'])
    b_s = int(pars['batch_size'])
    # choose pars for current stage of training
    # choose between shiny / normal

    rn = np.random.randint(low=-pars['seq_jitter'], high=pars['seq_jitter'])
    n_restart = np.maximum(n_restart - pars['curriculum_steps'], pars['restart_min'])

    pars['shiny_bias_env'] = [(0, 0) for _ in range(n_envs)]
    pars['direc_bias_env'] = [0 for _ in range(n_envs)]

    pars['shiny_sense'], shiny_s = choose_shiny_sense(pars)

    # make choice for each env
    choices = []
    for env in range(n_envs):
        choice = np.random.choice(pars['poss_behaviours'])

        choices.append(choice)

        if choice == 'shiny':
            pars['shiny_bias_env'][env] = pars_orig['shiny_bias']
        elif choice == 'normal':
            pars['direc_bias_env'][env] = pars_orig['direc_bias']
        else:
            raise Exception('Not a correct possible behaviour')

    # shiny_s for each batch
    for batch in range(b_s):
        env = pars['diff_env_batches_envs'][batch]
        choice = choices[env]
        if choice == 'normal':
            shiny_s[batch, :] = 0

    # choose which of batch gets no_direc or not - 1 is no_direc, 0 is with direc
    no_direc_batch = np.ones(pars['batch_size'])
    for batch in range(b_s):
        env = pars['diff_env_batches_envs'][batch]
        choice = choices[env]
        if choice == 'normal':
            no_direc_batch[batch] = 0
        else:
            no_direc_batch[batch] = 1

    return pars, shiny_s, rn, n_restart, no_direc_batch

def choose_shiny_sense(pars):

    # choose number of shiny objects per environment + choose which sensory stimuli will be shiny
    shiny_sense = [np.random.randint(0, pars['n_shiny_senses'], np.random.choice(pars['n_shiny']))
                   for _ in pars['widths']]

    # make mask for model - different for each batch
    shiny_s = np.zeros((pars['batch_size'], pars['s_size']))
    for i, s_s_env in enumerate(shiny_sense):
        for j, s_s_ in enumerate(s_s_env):
            shiny_s[pars['diff_env_batches_envs'][i], s_s_] = 1

    return shiny_sense, shiny_s

def get_walking_data(start_state, adj, tran, prev_d, shiny_states, n_walk, params):
    b_s = int(params['batch_size'])

    pos, d = np.zeros((b_s, n_walk + 1)), np.zeros((b_s, params['n_actions'], n_walk))

    for b in range(b_s):
        env = params['diff_env_batches_envs'][b]
        s_s = cp.deepcopy(shiny_states[env])

        pos[b, :], d[b, :, :], prev_d[b] = walk_square(adj[env], tran[env], n_walk, start_state[b], prev_d[b], s_s,
                                                           env, params)


    return pos, d

def get_new_data_diff_envs(position, data_envs, envs, states_mat, params):
    b_s = int(params['batch_size'])
    n_walk = params['n_walk']
    n_senses = params['n_senses']
    s_size = params['s_size']

    data = np.zeros((b_s, s_size, n_walk + 1))
    for batch in range(b_s):
        env = envs[batch]

        data[batch] = sample_data(position[batch, :], states_mat[env], n_senses, data_envs[batch])

    return data

def sample_data(position, states_mat, n_senses, last_data):

    time_steps = np.shape(position)[0]
    sense_data = np.zeros((n_senses[0], time_steps))
    sense_data[:, 0] = last_data[:, -1]
    for i, pos in enumerate(position):
        if i > 0:
            ind = int(pos)
            sense_data[int(states_mat[ind, 0]), i] = 1
    return sense_data
