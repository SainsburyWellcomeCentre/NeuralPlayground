import copy
import os
import pickle
import sys

import numpy as np

from neuralplayground.agents.george_2021_extras.george_2021_model import (
    CHMM,
    backtrace,
    backward,
    forward,
    forward_mp,
    forwardE,
    updateC,
)

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
        super().__init__(agent_name=agent_name, **mod_kwargs)
        self.mod_kwargs = mod_kwargs.copy()
        # params = mod_kwargs["params"]
        self.params = copy.deepcopy(mod_kwargs["params"])

        self.n_observations = mod_kwargs["n_observations"]
        self.n_actions = self.params["n_actions"]
        # we want to use a default if missing.
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
        self.pos_history = []
        self.global_steps = 0
        self.episode_history = []
        self.current_episode = {"obs": [], "act": [], "pos": []}

        self.n_clones_array = np.array([self.params["n_clones_per_obs"]] * self.n_observations, dtype=np.int64)

        # The CHMM class validates input sequences immediately upon __init__.
        # It requires arrays of int64. We provide a dummy sequence of zeros
        # so the object can be instantiated without crashing.
        dummy_x = np.zeros(2, dtype=np.int64)
        dummy_a = np.zeros(2, dtype=np.int64)
        dummy_a[-1] = self.n_actions - 1

        # print("x:", dummy_x.shape, "a:", dummy_a.shape)

        self.chmm = CHMM(
            n_clones=self.n_clones_array,  # uses the array we made in __init__
            x=dummy_x,
            a=dummy_a,
            pseudocount=self.params["pseudocount"],
            dtype=self.params["dtype"],
            seed=self.params["seed"],
        )

    def new_episode(self):
        """
        when env.reset() happens, we want to start a new episode in the agent's history.
        splits the history to prevent 'wormhole' transitions between episodes.
        """
        if len(self.current_episode["obs"]) > 0:
            # save the completed episode
            self.episode_history.append(self.current_episode)
        # start again
        self.current_episode = {"obs": [], "act": [], "pos": []}

    def obs_transformer_int(self, obs):
        """
        The CHMM model expects discrete integer observations, but the environment may provide them in various formats:
          - A single integer (e.g., 3)
          - A one-hot vector (e.g., [0, 0, 1, 0])
          - A list containing metadata (e.g., [index, object_vector, pos])
        """

        if isinstance(obs, (list, np.ndarray)):
            if len(obs) >= 2:  # discreteobjenv returns [index, object (vector), pos]
                object_data = obs[1]

                if isinstance(object_data, (list, np.ndarray)):
                    return int(np.argmax(object_data))
                    # index of the "1" within the one-hot-vector

                else:
                    return int(object_data)

        # fallback
        return int(obs)

        raise ValueError(f"Unknown observation format: {type(obs)}")

    def act(self, obs, policy_func=None):
        """
        processes an observation and returns an action vector
        """
        if len(obs) == 0:
            return None

        discrete_obs = self.obs_transformer_int(obs)

        if policy_func is not None:
            discrete_action = policy_func(obs)
        else:
            discrete_action = np.random.randint(0, self.n_actions)

        self.obs_history.append(discrete_obs)
        self.action_history.append(discrete_action)

        self.current_episode["obs"].append(discrete_obs)
        self.current_episode["act"].append(discrete_action)

        self.global_steps += 1

        return self.action_translation(discrete_action)

    def update(self):
        """
        Perform model update (EM or Viterbi) using accumulated history.
        Copies logic from CHMM.learn_em_T and CHMM.learn_viterbi_T but
        iterates over episodes to avoid 'wormholes'.
        """
        # Collect all data
        # training_data = self.episode_history + [self.current_episode]
        training_data = self.episode_history

        total_steps = sum(len(ep["obs"]) for ep in training_data)

        # Check batch size
        if total_steps < self.batch_size:
            return None

        # User chooses algo in params: "EM" or "Viterbi"
        learning_algo = self.params.get("learning_algo", "EM")
        n_iter = self.params["n_iterations"]
        term_early = self.params.get("term_early", True)

        log_lik_history = []

        # Main Training Loop
        for i in range(n_iter):
            self.chmm.C[:] = 0  # Reset global counts accumulator for this iteration of EM
            total_log_lik = 0

            # Iterate over episodes to accumulate counts (E-Step equivalent)
            for ep in training_data:
                if len(ep["obs"]) < 2:
                    continue

                x = np.array(ep["obs"], dtype=np.int64)
                a = np.array(ep["act"], dtype=np.int64)

                # learning algorithm set by user
                if learning_algo == "Viterbi":
                    # VITERBI
                    # adapted from CHMM.learn_viterbi_T

                    log2_lik, mess_fwd = forward_mp(
                        self.chmm.T.transpose(0, 2, 1), self.chmm.Pi_x, self.chmm.n_clones, x, a, store_messages=True
                    )

                    states = backtrace(self.chmm.T, self.chmm.n_clones, x, a, mess_fwd)

                    for t in range(1, len(x)):
                        aij = a[t - 1]
                        prev_state = states[t - 1]
                        curr_state = states[t]
                        self.chmm.C[aij, prev_state, curr_state] += 1.0

                    total_log_lik += log2_lik.mean()

                else:
                    # EM
                    # adapted from CHMM.learn_em_T

                    log2_lik, mess_fwd = forward(
                        self.chmm.T.transpose(0, 2, 1), self.chmm.Pi_x, self.chmm.n_clones, x, a, store_messages=True
                    )

                    mess_bwd = backward(self.chmm.T, self.chmm.n_clones, x, a)

                    # temporary buffer
                    episode_C = np.zeros_like(self.chmm.C)

                    # updateC will reset 'episode_C' to zero, then fill it with this episode's counts
                    updateC(episode_C, self.chmm.T, self.chmm.n_clones, mess_fwd, mess_bwd, x, a)

                    # Now we accumulate this episode's counts into the main batch accumulator
                    self.chmm.C += episode_C

                    total_log_lik += log2_lik.sum()

            self.chmm.update_T()

            log_lik_history.append(total_log_lik)

            # check for convergence
            if term_early and i > 0:
                diff = log_lik_history[-1] - log_lik_history[-2]
                if abs(diff) < 1e-4:
                    break

        # self.global_steps += 1

        # Clear episode history after a successful batch update
        self.episode_history = []

        return {"log_lik": log_lik_history[-1] if log_lik_history else 0}

    # ... (Keep existing __init__, reset, act, update, save_agent methods) ...

    def mess_fwd(self):
        """
        Calculates forward messages (belief states) for the entire recorded history.

        Replicates the colab's get_mess_fwd approach:
          - Rebuilds T from C with a small pseudocount so zero-transitions don't crash.
          - Averages T over actions, making belief propagation action-independent and
            robust. This matches the colab's use of T.mean(0) and x*0 for actions.
          - Processes episodes individually to avoid wormhole artifacts.
        """
        all_messages = []

        n_clones = self.chmm.n_clones

        # Deterministic emission matrix: clone i → observation obs_map[i]
        pseudocount_E = 0.0001

        E = np.zeros((n_clones.sum(), len(n_clones)))
        last = 0
        for c in range(len(n_clones)):
            E[last : last + n_clones[c], c] = 1
            last += n_clones[c]
        E += pseudocount_E
        norm = E.sum(1, keepdims=True)
        norm[norm == 0] = 1
        E /= norm

        pseudocount = 0.0  # small enough not to distort learned structure
        T = self.chmm.C.astype(np.float64) + pseudocount
        norm = T.sum(2, keepdims=True)
        norm[norm == 0] = 1
        T /= norm
        T_avg = T.mean(0, keepdims=True)  # shape (1, n_states, n_states)
        T_avg_trans = T_avg.transpose(0, 2, 1).astype(self.chmm.C.dtype)

        for ep in self.episode_history + [self.current_episode]:
            if len(ep["obs"]) == 0:
                continue

            x = np.array(ep["obs"], dtype=np.int64)
            a = np.zeros(len(x), dtype=np.int64)  # dummy actions for action-averaged T

            _, mess = forwardE(T_avg_trans, E, self.chmm.Pi_x, n_clones, x, a, store_messages=True)
            all_messages.append(mess)

        if not all_messages:
            return np.array([])

        return np.concatenate(all_messages, axis=0)

    def get_belief_state(self):
        return self.mess_fwd()

    def save_agent(self, save_path: str, raw_object: bool = True):
        """
        Saves current state and parameters to a file.
        """
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_data = {
            "chmm": self.chmm,
            "obs_history": self.obs_history,
            "action_history": self.action_history,
            "pos_history": self.pos_history,  # ← was missing
        }

        with open(save_path, "wb") as f:
            pickle.dump(save_data, f, pickle.HIGHEST_PROTOCOL)

        param_path = os.path.join(save_dir, "chmm_agent_params.pkl")
        with open(param_path, "wb") as f:
            pickle.dump(self.params, f, pickle.HIGHEST_PROTOCOL)

        print(f"agent saved to {save_path}")
