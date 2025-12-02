import copy
import os
import pickle
import sys

import numpy as np

from neuralplayground.agents.george_2021_extras.george_2021_model import CHMM

from .agent_core import AgentCore

sys.path.append("../")


class George2021(AgentCore):
    """
    Implementation of CSCG 2021 by Dileep George, Rajeev V. Rikhye, Nishad Gothoskar, J. Swaroop Guntupalli,
    Antoine Dedieu & Miguel Lázaro-Gredilla. Clone-structured graph representations enable flexible learning and
    vicarious evaluation of cognitive maps. https://doi.org/10.1038/s41467-021-22559-5
    ----
    Attributes
    ---------
    mod_kwargs : dict
        Model parameters
        params: dict
            contains the majority of parameters used by the model and environment
        n_observations: int
            total number of unique discrete states in the environment
        n_actions: int
            total number of unique discrete actions available
    chmm : class
        The underlying Clone-Structured Hidden Markov Model (CHMM) object

    Methods
    ---------
    reset(self):
        initialise model and associated variables for training
    act(self, obs, policy_func):
        Processes an observation and returns an action vector. If policy_func is None,
        selects a random action.
    update(self):
        Perform model update (EM or Viterbi training) using accumulated history and return convergence stats
    action_translation(self):
        Helper method to define action mappings between discrete model integers and arena vectors. Converts an
        integer action index to a physical movement vector
    save_agent(self, save_path):
        Save current state and parameters to a file

    """

    # initialises hyperparameters such as n_clones, batch_size etc
    def __init__(self, agent_name: str = "CSCG", **mod_kwargs):
        """
        Parameters
        ----------
        model_name : str
           Name of the specific instantiation of the ExcInhPlasticity class
        mod_kwargs : dict
            params: dict
                contains the majority of parameters used by the model and environment
            n_observations: int
                total number of unique states in the environment
            n_actions: int
                total number of unique actions agent can perform
        """
        super().__init__(agent_name=agent_name, **mod_kwargs)
        self.mod_kwargs = mod_kwargs.copy()
        # params = mod_kwargs["params"]
        self.params = copy.deepcopy(mod_kwargs["params"])

        self.n_observations = mod_kwargs["n_observations"]
        self.n_actions = self.params["n_actions"]
        # We want this to be polite and use a default if missing.
        self.batch_size = mod_kwargs["batch_size"]

        self.action_map = [np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0])]

        self.chmm = None

        self.reset()

    def action_translation(self, discrete_action):
        """
        the cscg model views actions in integers (0,1,2,3 etc) but neural playground
        arena requires movement vectors ([0, 1], [1, 0] etc).

        this translates actions using the action map in __init__.
        """

        return self.action_map[discrete_action]

    def reset(self):
        """
        initialise model and associated variables for training
        resets history buffers and re-instantiates the CHMM class
        """
        # model is initialised here instead of in __init__
        # this is because it needs an array of obs, actions (x, a)
        self.obs_history = []
        self.action_history = []
        self.global_steps = 0

        self.n_clones_array = np.array([self.params["n_clones_per_obs"]] * self.n_observations, dtype=np.int64)

        # The CHMM class validates input sequences immediately upon __init__.
        # It requires arrays of int64. We provide a dummy sequence of zeros
        # so the object can be instantiated without crashing.
        dummy_x = np.zeros(2, dtype=np.int64)
        dummy_a = np.zeros(2, dtype=np.int64)

        self.chmm = CHMM(
            n_clones=self.n_clones_array,  # Use the array we made in __init__
            x=dummy_x,
            a=dummy_a,
            pseudocount=self.params["pseudocount"],
            dtype=self.params["dtype"],
            seed=self.params["seed"],
        )

    def act(self, obs, policy_func=None):
        """
        processes an observation and returns an action vector
        """
        if len(obs) == 0:
            return None
        discrete_obs = int(obs[0])

        if policy_func is not None:
            discrete_action = policy_func(obs)
        else:
            discrete_action = np.random.randint(0, self.n_actions)

        self.obs_history.append(discrete_obs)
        self.action_history.append(discrete_action)

        return self.action_translation(discrete_action)

    def update(self):
        """
        perform model update (EM or Viterbi training) using accumulated history
        and return convergence stats
        """
        if len(self.obs_history) < self.batch_size:
            return None

        x_train = np.array(self.obs_history, dtype=np.int64)
        a_train = np.array(self.action_history, dtype=np.int64)

        learning_algo = self.params["learning_algo"]
        n_iter = self.params["n_iterations"]

        if learning_algo == "EM":
            # Expectation-Maximization
            # term_early allows stopping if log-likelihood converges
            term_early = self.params.get("term_early", True)
            convergence = self.chmm.learn_em_T(x_train, a_train, n_iter=n_iter, term_early=term_early)
        elif learning_algo == "Viterbi":
            # Viterbi Training (Hard EM)
            convergence = self.chmm.learn_viterbi_T(x_train, a_train, n_iter=n_iter)
        else:
            print(f"Unknown learning algorithm: {learning_algo}")

        self.global_steps += 1

        return convergence

    def save_agent(self, save_path: str, raw_object: bool = True):
        """
        saves current state and parameters to a file
        """
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save the CHMM object (contains learned T and E matrices)
        with open(save_path, "wb") as f:
            pickle.dump(self.chmm, f, pickle.HIGHEST_PROTOCOL)

        # Save the hyperparameters separately for easier inspection
        param_path = os.path.join(save_dir, "chmm_agent_params.pkl")
        with open(param_path, "wb") as f:
            pickle.dump(self.params, f, pickle.HIGHEST_PROTOCOL)

        print(f"Agent saved to {save_path}")
