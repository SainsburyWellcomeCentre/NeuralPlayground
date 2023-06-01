import torch
import numpy as np
import copy as cp
import os
import datetime
import logging

def inv_var_weight(mus, sigmas):
    '''
    Accepts lists batches of row vectors of means and standard deviations, with batches along dim 0
    Return tensors of inverse-variance weighted averages and tensors of inverse-variance weighted standard deviations
    '''
    # Stack vectors together along first dimension
    mus = torch.stack(mus, dim = 0)
    sigmas = torch.stack(sigmas, dim = 0)
    # Calculate inverse variance weighted variance from sum over reciprocal of squared sigmas
    inv_var_var = 1.0 / torch.sum(1.0 / (sigmas**2), dim = 0)
    # Calculate inverse variance weighted average
    inv_var_avg = torch.sum(mus / (sigmas**2), dim = 0) * inv_var_var
    # Convert weigthed variance to sigma
    inv_var_sigma = torch.sqrt(inv_var_var)
    # And return results
    return inv_var_avg, inv_var_sigma

def softmax(x):
    '''
    Applies softmax to tensors of inputs, using torch softmax funcion
    Assumes x is a 1D vector, or batches of row vectors with the batches along dim 0
    '''
    # Return torch softmax
    return torch.nn.Softmax(dim=-1)(x)

def normalise(x):
    '''
    Normalises vector of input to unit norm, using torch normalise funcion
    Assumes x is a 1D vector, or batches of row vectors with the batches along dim 0
    '''
    # Return torch normalise with p=2 for L2 norm
    return torch.nn.functional.normalize(x, p=2, dim=-1)

def relu(x):
    '''
    Applies rectified linear activation unit to tensors of inputs, using torch relu funcion
    '''
    # Return torch relu
    return torch.nn.functional.relu(x)

def leaky_relu(x):
    '''
    Applies leaky (meaning small negative slope instead of zeros) rectified linear activation unit to tensors of inputs, using torch leaky relu funcion
    '''
    # Return torch leaky relu [torch.nn.functional.leaky_relu(val) for val in x] if type(x) is list else
    return torch.nn.functional.leaky_relu(x)

def squared_error(value, target):
    '''
    Calculates mean squared error (L2 norm) between (list of) tensors value and target by using torch MSE loss
    Include a factor 0.5 to squared error by convention
    Set reduction to none, then get mean over last dimension to keep losses of different batches separate
    '''
    # Return torch MSE loss
    if type(value) is list:
        loss = [0.5 * torch.sum(torch.nn.MSELoss(reduction='none')(value[i], target[i]),dim=-1) for i in range(len(value))]
    else:
        loss = 0.5 * torch.sum(torch.nn.MSELoss(reduction='none')(value, target),dim=-1)
    return loss

def cross_entropy(value, target):
    '''
    Calculates binary cross entropy between tensors value and target by using torch cross entropy loss
    Set reduction to none, then get mean over last dimension to keep losses of different batches separate
    '''
    # Return torch BCE loss
    if type(value) is list:
        loss = [torch.nn.CrossEntropyLoss(reduction='none')(val, targ) for val, targ in zip(value, target)]
    else:
        loss = torch.nn.CrossEntropyLoss(reduction='none')(value, target)
    return loss

def downsample(value, target_dim):
    '''
    Does downsampling by taking the an input vector, then averaging chunks to make it of requested dimension
    Assumes x is a 1D vector, or batches of row vectors with the batches along dim 0
    '''
    # Get input dimension
    value_dim = value.size()[-1]
    # Set places to break up input vector into chunks
    edges = np.append(np.round(np.arange(0, value_dim, float(value_dim) / target_dim)),value_dim).astype(int)
    # Create downsampling matrix
    downsample = torch.zeros((value_dim,target_dim), dtype = torch.float)
    # Fill downsampling matrix with chunks
    for curr_entry in range(target_dim):
        downsample[edges[curr_entry]:edges[curr_entry+1],curr_entry] = torch.tensor(1.0/(edges[curr_entry+1]-edges[curr_entry]), dtype=torch.float)
    # Do downsampling by matrix multiplication
    return torch.matmul(value,downsample)

def make_directories():
    '''
    Creates directories for storing data during a model training run
    '''
    # Get current date for saving folder
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    # Initialise the run and dir_check to create a new run folder within the current date
    run = 0
    dir_check = True
    # Initialise all pahts
    train_path, model_path, save_path, script_path, run_path = None, None, None, None, None
    # Find the current run: the first run that doesn't exist yet
    while dir_check:
        # Construct new paths
        run_path = '../Summaries2/' + date + '/torch_run' + str(run) + '/'
        train_path = run_path + 'train'
        model_path = run_path + 'model'
        save_path = run_path + 'save'
        script_path = run_path + 'script'
        envs_path = script_path + '/envs'
        run += 1
        # And once a path doesn't exist yet: create new folders
        if not os.path.exists(train_path) and not os.path.exists(model_path) and not os.path.exists(save_path):
            os.makedirs(train_path)
            os.makedirs(model_path)
            os.makedirs(save_path)
            os.makedirs(script_path)
            os.makedirs(envs_path)
            dir_check = False
    # Return folders to new path
    return run_path, train_path, model_path, save_path, script_path, envs_path

def set_directories(date, run):
    '''
    Returns directories for storing data during a model training run from a given previous training run
    '''
    # Initialise all pahts
    train_path, model_path, save_path, script_path, run_path = None, None, None, None, None
    # Find the current run: the first run that doesn't exist yet
    run_path = '../Summaries/' + date + '/run' + str(run) + '/'
    train_path = run_path + 'train'
    model_path = run_path + 'model'
    save_path = run_path + 'save'
    script_path = run_path + 'script'
    envs_path = script_path + '/envs'
    # Return folders to new path
    return run_path, train_path, model_path, save_path, script_path, envs_path

def make_logger(run_path):
    '''
    Creates logger so output during training can be stored to file in a consistent way
    '''
    # Create new logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Remove anly existing handlers so you don't output to old files, or to new files twice
    logger.handlers = []
    # Create a file handler, but only if the handler does
    handler = logging.FileHandler(run_path + 'report.log')
    handler.setLevel(logging.INFO)
    # Create a logging format
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    handler.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(handler)
    # Return the logger object
    return logger

# Function used for saving torch variables as in TensorFlow version
def prepare_data_maps(data, prev_cell_maps, positions, pars):
    gs, ps, position = data
    gs_all, ps_all = prev_cell_maps

    g1s = np.transpose(np.array(cp.deepcopy(gs)), [1, 2, 0])
    p1s = np.transpose(np.array(cp.deepcopy(ps)), [1, 2, 0])
    # pos_to = position[:][1:pars['n_rollout'] + 1]
    pos_to = position

    gs_all = cell_norm_online(g1s, pos_to, gs_all, pars)
    ps_all = cell_norm_online(p1s, pos_to, ps_all, pars)

    cell_list = [gs_all, ps_all]

    return cell_list, positions

def cell_norm_online(cells, positions, current_cell_mat, pars):
    # for separate environments within each batch
    envs = pars['diff_env_batches_envs']
    n_states = pars['n_states_world']
    n_envs_save = pars['n_envs_save']

    num_cells = np.shape(cells)[1]
    n_trials = np.shape(cells)[2]

    cell_mat = [np.zeros((n_states[envs[env]], num_cells)) for env in range(n_envs_save)]

    new_cell_mat = [None] * n_envs_save

    for env in range(n_envs_save):
        for ii in range(n_trials):
            position = int(positions[ii][env]['id'])
            cell_mat[env][position, :] += cells[env, :, ii]
        try:
            new_cell_mat[env] = cell_mat[env] + current_cell_mat[env]
        except:
            new_cell_mat[env] = cell_mat[env]

    return new_cell_mat

def check_wall(pre_state, new_state, wall, wall_closenes=1e-5, tolerance=1e-9):
    """

    Parameters
    ----------
    pre_state : (2,) 2d-ndarray
        2d position of pre-movement
    new_state : (2,) 2d-ndarray
        2d position of post-movement
    wall : (2, 2) ndarray
        [[x1, y1], [x2, y2]] where (x1, y1) is on limit of the wall, (x2, y2) second limit of the wall
    wall_closenes : float
        how close the agent is allowed to be from the wall

    Returns
    -------
    new_state: (2,) 2d-ndarray
        corrected new state. If it is not crossing the wall, then the new_state stays the same, if the state cross the
        wall, new_state will be corrected to a valid place without crossing the wall
    cross_wall: bool
        True if the change in state cross a wall
    """

    # Check if the line of the wall and the line between the states cross
    A = np.stack([np.diff(wall, axis=0)[0, :], -new_state + pre_state], axis=1)
    b = pre_state - wall[0, :]
    try:
        intersection = np.linalg.inv(A) @ b
    except:
        intersection = np.linalg.inv(A + np.identity(A.shape[0]) * tolerance) @ b
    smaller_than_one = intersection <= 1
    larger_than_zero = intersection >= 0

    # If condition is true, then the points cross the wall
    cross_wall = np.alltrue(np.logical_and(smaller_than_one, larger_than_zero))
    if cross_wall:
        new_state = (intersection[-1] - wall_closenes) * (new_state - pre_state) + pre_state

    return new_state, cross_wall
