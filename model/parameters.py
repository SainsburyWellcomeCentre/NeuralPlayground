import numpy as np
from copy import deepcopy
#######################################################################
############################## Parameters #############################
#######################################################################
# Define the parameters for the simulation in a dictionary
# The structure is:
# 'exc' / 'inh': For excitatory and inhibitory synapses
# 'sim': For main simulation parameters
# 'out': For parameters that have to do with the output neurons

def modify_parameters(params_dict, modifications):
    """
    Convenience function to modifiy an existing parameters dictionary

    Parameters
    ----------
    params_dict : dict
        Dictionary with simulation parameters
    modifications : list of 3-tuples
        Each 3 tuple
        Example: [('exc', 'sigma', np.array

    Returns
    -------
    prms : dict
        A complete parameters dictionary with the specified modifications
    """
    prms = deepcopy(params_dict)
    for m in modifications:
        prms[m[0]][m[1]] = m[2]
    return prms

# Different example sets

###############################################
######### 1D, place cell to grid cell #########
###############################################
# Specifying the tuning width and the simulation time outside is useful,
# because we make some parameters depend on it
sigma_exc = np.array([0.04])
sigma_inh = np.array([0.13])
simulation_time = 4e4
input_space_resolution = sigma_exc / 10.
every_nth_step = simulation_time / 100
params_1d_place2grid = {
    'subdimension': 'none',
    'sim':
        {
            # The simulation time
            'simulation_time': simulation_time,
            # Spacing of the stored rate maps (along each dimension)
            'spacing': 201,
            # Number of dimensions
            'dimensions': 1,
            # Boxtype, either linear (with edges) or circular
            'boxtype': 'linear',
            # Radius
            'radius': 1,
            # Every n_th step of data is stored
            'every_nth_step': every_nth_step,
            # Same as above, but for the synaptic weights
            'every_nth_step_weights': every_nth_step,
            # Seeds the random initial weights
            'seed_init_weights': 0,
            # Seeds the randomness in the input tuning
            'seed_centers': 0,
            # Seeds the random tuning widths of different input neurons
            'seed_sigmas': 0,
            # Seeds the random motion
            'seed_motion': 0,
            # Initial positions (must be within -radius, radius)
            'initial_x': 0.1,
            'initial_y': 0.2,
            'initial_z': 0.15,
            # Persistence length of persistent random walk
            'persistence_length': 1,
            # Type of motion
            'motion': 'persistent',
            # Whether or not input tuning is gaussian random field
            'gaussian_process': False,
            # How to rescale the Gaussian random fields
            'gaussian_process_rescale': 'fixed_mean',
            # Resolution of space discretization
            'input_space_resolution': input_space_resolution,
            #################################################
            ### Parameters that are typically not changed ###
            #################################################
            # Rat velocity
            'velocity': 1e-2,
            # The unit time step
            'dt': 1,
            # Diffusion constant for diffusive motion
            'diff_const': 0.01,
            # Rat that doesn't move
            'stationary_rat': False,
            # Same centers for excitatory and inhibitory inputs
            'same_centers': False,
            # Center of input firing fields chose from distorted lattice
            'symmetric_centers': True,
            # Choose initial weight close to fixed point at target rate
            'take_fixed_point_weights': True,
            # Wether there is an input neuron with firing field at origin
            'first_center_at_zero': False,
            # Type of elementary tuning function
            'tuning_function': 'gaussian',
            # Store 2 sigma**2 array (not important)
            'store_twoSigma2': False,
            # Numper of input tuning examples that are stored
            # NB: It can also be specified for each synapse type
            # individually. Then set it False here.
            'save_n_input_rates': 3,
            # Normalize each input rate
            'input_normalization': 'none',
            # Noisy alignment of head direction with running direction
            'head_direction_sigma': np.pi / 6.,
            # Discretize space for efficiancy
            'discretize_space': True,
            # Convolution dx (not important)
            'fixed_convolution_dx': False,
            # Scaling the excitatory weights with the variance of the
            #  tuning functions.
            'scale_exc_weights_with_input_rate_variance': False,
            #######################################################
            ### Switching box side (wall experiment) parameters ###
            #######################################################
            'boxside_independent_centers': False,
            # The boxside in which the rat learns first, for the
            # boxside switch experiments.
            'boxside_initial_side': 'left',
            # Time at which the rat can explore the entire arena
            # Set to False, if no wall experiment is conducted.
            'explore_all_time': False,
            # Time at which the rat should switch to the right side
            # of the box on move only in the right side.
            # Set to False, if no wall experiment is conducted.
            'boxside_switch_time': False,
            # We typically do not start in room2, so default is False
            'in_room2': False,
            # Correlation between the inputs
            'alpha_room1': 1,
            'alpha_room2': 0.5,
            'room_switch_method': 'some_inputs_identical',
            'room_switch_time': False,
            #####################################################
            ### Parameters relevant only for recurrent system ###
            #####################################################
            # Equilibration steps (for the recurrent system only)
            'equilibration_steps': 10000,
            # Lateral inhibition (for the recurrent system only)
            'lateral_inhibition': False,
            # Number of output neurons (for the recurrent system only)
            'output_neurons': 1,
            # Weight between output neurons (for the recurrent system only)
            'weight_lateral': 0.0,
            # Neural time constant (for the recurrent system only)
            'tau': 10.,
        },
    'out':
        {
            # Target rate of the output neuron
            'target_rate': 1,
            # Method of weight normalization
            'normalization': 'quadratic_multiplicative',
        },
    'exc':
        {   # Number of input rate examples to be saved
            'save_n_input_rates': 3,
            # Factor of how many standard deviations the field centers should
            #  be outside the box
            'center_overlap_factor': 3.,
            # Number of inputs along each dimension
            'number_per_dimension': np.array([160]),
            # Learning rate
            'eta': 1e-3,
            # Number of place fields per input neuron
            'fields_per_synapse': 1,
            # Width of the input tuning, i.e., smoothness.
            'sigma': sigma_exc,
            # Spreading over the width
            'sigma_spreading': np.array([0]),
            # Distribution of the widths
            'sigma_distribution': np.array(['uniform']),
            # Mean of initial synaptic weight
            'init_weight': 1,
            # Spreading of the initial weights
            'init_weight_spreading': 5e-2,
            # Distribution of the initial weights
            'init_weight_distribution': 'uniform',
            # The height of the Gaussians
            # NB: Only meaningful if no input_normalization is 'none'
            'gaussian_height': 1,
            'real_gaussian_height': 1,
            # For untuned (constant) input
            'untuned': False,
            # Distortion of the lattice on which the input centers are arranged
            'distortion': 'half_spacing',
            # If grid cells are use as input
            'grid_input_sidelength': 10,
            'grid_input_spacing_noise': 0.1,
            # If Gaussian process should be stretched or of fixed extremum
            'gp_stretch_factor': 1.0,
            'gp_extremum': 'none',
        },
    'inh':
        {
            # For all parameters see description above for 'exc'.
            'grid_input_sidelength': 1,
            'grid_input_spacing_noise': 0.,
            'save_n_input_rates': 3,
            'gp_stretch_factor': 1.0,
            'gp_extremum': 'none',
            'center_overlap_factor': 3.,
            'weight_factor': 1,
            'number_per_dimension': np.array([40]),
            'distortion': 'half_spacing',
            'eta': 1e-2,
            'sigma': sigma_inh,
            'sigma_spreading': np.array([0]),
            'sigma_distribution': np.array(['uniform']),
            'fields_per_synapse': 1,
            'init_weight': 1.0,
            'init_weight_spreading': 5e-2,
            'init_weight_distribution': 'uniform',
            'gaussian_height': 1,
            'real_gaussian_height': 1,
            'untuned': False,
        }
}

###########################################################################
##################### 1D, non-localized to place cell #####################
###########################################################################
sigma_exc = np.array([0.06])
simulation_time = 4e5
params_1d_non_localized2place = modify_parameters(
        params_1d_place2grid,
    [
        ('sim', 'simulation_time', simulation_time),
        ('sim', 'every_nth_step', simulation_time / 100),
        ('sim', 'every_nth_step_weights', simulation_time / 100),
        ('sim', 'input_space_resolution', sigma_exc / 10.),
        ('exc', 'sigma', sigma_exc),
        ('sim', 'gaussian_process', True),
        ('sim', 'tuning_function', 'gaussian_process'),
        ('inh', 'untuned', True),
        ('exc', 'number_per_dimension', np.array([2000])),
        ('inh', 'number_per_dimension', np.array([500])),
        ('exc', 'eta', 2e-6),
        ('inh', 'eta', 2e-5),
    ])

##########################################################################
##################### 1D, non-localized to invariant #####################
##########################################################################
sigma_exc = np.array([0.08])
sigma_inh = np.array([0.07])
params_1d_non_localized2invariant = modify_parameters(
    params_1d_non_localized2place,
    [
        ('inh', 'untuned', False),
        ('exc', 'sigma', sigma_exc),
        ('inh', 'sigma', sigma_inh),
        ('exc', 'eta', 2e-5),
        ('inh', 'eta', 2e-4),
        ('sim', 'input_space_resolution', sigma_exc / 10.),
    ])

#####################################################################
##################### 1D, non-localized to grid #####################
#####################################################################
sigma_exc = np.array([0.05])
sigma_inh = np.array([0.12])
params_1d_non_localized2grid = modify_parameters(
    params_1d_non_localized2invariant,
    [
        ('exc', 'sigma', sigma_exc),
        ('inh', 'sigma', sigma_inh),
        ('sim', 'input_space_resolution', sigma_exc / 10.),
    ])

##############################################
############### 1D: Quick test run ###########
##############################################
params_test = modify_parameters(
        params_1d_place2grid,
    [
        ('sim', 'simulation_time', 2),
        ('sim', 'spacing', 5),
        ('sim', 'every_nth_step', 1),
        ('sim', 'every_nth_step_weights', 1),
        ('exc', 'number_per_dimension', np.array([2])),
        ('inh', 'number_per_dimension', np.array([2])),
    ])


####################################
### 2D, place cell to grid cell  ###
####################################
sigma_exc = np.array([0.05, 0.05])
sigma_inh = np.array([0.10, 0.10])
simulation_time = 18e5
sigma_spreading = np.array([0, 0])
sigma_distribution = np.array(['uniform', 'uniform'])
params_2d_place2grid = modify_parameters(
    params_1d_place2grid,
    [
        ('sim', 'dimensions', 2),
        ('exc', 'sigma', sigma_exc),
        ('inh', 'sigma', sigma_inh),
        ('exc', 'eta', 6.7e-5),
        ('inh', 'eta', 2.7e-4),
        ('sim', 'simulation_time', simulation_time),
        ('exc', 'number_per_dimension', np.array([70, 70])),
        ('inh', 'number_per_dimension', np.array([35, 35])),
        ('sim', 'input_space_resolution', sigma_exc / 4.),
        ('sim', 'every_nth_step', simulation_time / 2),
        ('sim', 'every_nth_step_weights', simulation_time / 2),
        ('exc', 'sigma_spreading', sigma_spreading),
        ('inh', 'sigma_spreading', sigma_spreading),
        ('exc', 'sigma_distribution', sigma_distribution),
        ('inh', 'sigma_distribution', sigma_distribution),
        ('sim', 'radius', 0.5),
        ('sim', 'motion', 'sargolini_data'),
        ('sim', 'spacing', 51),
    ])

#######################################
### 2D, non-localized to grid cell  ###
#######################################
params_2d_non_localized2grid = modify_parameters(
    params_2d_place2grid,
    [
        ('exc', 'eta', 2e-6),
        ('inh', 'eta', 8e-6),
        ('exc', 'fields_per_synapse', 5),
        ('inh', 'fields_per_synapse', 5),
    ])

######################################
## 2D, non-localized to place cell  ##
######################################
params_2d_non_localized2grid = modify_parameters(
    params_2d_place2grid,
    [
        ('exc', 'eta', 2e-6),
        ('inh', 'eta', 8e-6),
        ('exc', 'fields_per_synapse', 100),
        ('inh', 'fields_per_synapse', 100),
    ])

##############################################
############### Quick test run 2d ############
##############################################
params_test_2d = modify_parameters(
        params_2d_place2grid,
    [
        ('sim', 'simulation_time', 2),
        ('sim', 'spacing', 5),
        ('sim', 'every_nth_step', 1),
        ('sim', 'every_nth_step_weights', 1),
        ('exc', 'number_per_dimension', np.array([2, 2])),
        ('inh', 'number_per_dimension', np.array([2, 2])),
    ])