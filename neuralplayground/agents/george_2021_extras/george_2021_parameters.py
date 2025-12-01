import numpy as np

def parameters():
    """
    Set all parameters for the Cloned HMM (CHMM) model. This is a function so that it can be called from other scripts,
        e.g. to load parameters from a file.
    """
    params = {}
    # -- Model parameters
    # The number of 'clones' (hidden states) to assign to each observation[cite: 54, 840].
    params["n_clones_per_obs"] = 20  # [cite: 893, 901, 910]
    # The pseudocount for smoothing the transition matrix (T).
    # Prevents zero probabilities.
    params["pseudocount"] = 1e-2  # [cite: 901, 910]
    # The 'extra' pseudocount used when learning the emission matrix (E).
    params["pseudocount_extra"] = 1e-20
    # Data type for numpy arrays in the model.
    params["dtype"] = np.float32
    # Random seed for reproducibility.
    params["seed"] = 42

    # -- Training parameters
    # Specify the learning algorithm to use in the update() step.
    # Can be 'EM' or 'Viterbi'. Set as EM by default.
    params["learning_algo"] = "EM"
    # Number of iterations for the EM or Viterbi algorithm to run during each update().
    params["n_iterations"] = 100 
    # Flag for EM to terminate early if log-likelihood doesn't improve.
    params["term_early"] = True

    # -- Environment/Data parameters
    # These are needed by the CHMMAgent to initialize the CHMM
    # object, as it requires info about the data shape.
    # Number of distinct observations (e.g., colors, textures) in the environment.
    params["n_observations"] = 50  # Placeholder: Change this value
    
    # Number of actions (e.g., 0:left, 1:right, 2:up, 3:down).
    params["n_actions"] = 4

    return params