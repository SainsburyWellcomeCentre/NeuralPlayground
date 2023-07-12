"""
Base class for models that can interact with environments in this repo
Any neuralplayground model should inherit this class in order to interact with
environments and compare against experimental results.
We expect to make profound changes in this module as we add more EHC model to the repo
"""
import os
import pickle

import numpy as np
import pandas as pd
from deepdiff import DeepDiff
from scipy.stats import levy_stable


class AgentCore(object):
    """Abstract class for all EHC models

    Attributes
    ----------
    model_name : str
        Name of the specific instantiation of the ExcInhPlasticity class
    mod_kwargs: dict
        Dictionary of specific parameters to be used by children classes
    metadata
        Specific data structure which will contain specific description for each model
    obs_history: list
        List of past observations while interacting with the environment in the act method
    global_steps: int
        Record of number of updates done on the weights
    """

    def __init__(self, model_name="default_model", **mod_kwargs):
        self.model_name = model_name
        self.mod_kwargs = mod_kwargs
        if "agent_step_size" in mod_kwargs.keys():
            self.agent_step_size = mod_kwargs["agent_step_size"]
        else:
            self.agent_step_size = 1.0
        self.metadata = {"mod_kwargs": mod_kwargs}
        self.obs_history = []
        self.global_steps = 0

    def reset(self):
        """Erase all memory from the model, initialize all relevant parameters and build from scratch"""
        pass

    def act(self, obs, policy_func=None):
        """
        The base model executes a random action from a normal distribution
        Parameters
        ----------
        obs
            Observation from the environment class needed to choose the right action
        policy_func
            Arbitrary function that represents a custom policy that receives and observation and gives an action
        Returns
        -------
        action: float
            action value which in this case is random number draw from a Gaussian
        """
        self.obs_history.append(obs)
        if len(self.obs_history) >= 1000:  # reset every 1000
            self.obs_history = [
                obs,
            ]
        if policy_func is not None:
            return policy_func(obs)
        action = np.random.normal(scale=self.agent_step_size, size=(2,))
        return action

    def update(self):
        pass

    def save_agent(self, save_path: str):
        """Save current state and information in general to re-instantiate the environment

        Parameters
        ----------
        save_path: str
            Path to save the agent
        """
        # pickle.dump(self.__dict__, open(os.path.join(save_path), "wb"), pickle.HIGHEST_PROTOCOL)
        pickle.dump(self, open(os.path.join(save_path), "wb"), pickle.HIGHEST_PROTOCOL)

    def restore_agent(self, save_path: str):
        """Restore saved environment

        Parameters
        ----------
        save_path: str
            Path to retrieve the agent
        """
        # self.__dict__ = pd.read_pickle(save_path)
        # TODO: for some reason, ruff has a problem with this: self = pd.read_pickle(save_path)
        pd.read_pickle(save_path)

    def __eq__(self, other):
        diff = DeepDiff(self.__dict__, other.__dict__)
        if len(diff) == 0:
            return True
        else:
            return False

    def get_ratemap_matrix(self):
        """Function that returns some representation that will be compared against real experimental data"""
        pass


class RandomAgent(AgentCore):
    """Simple agent with random trajectories"""

    def __init__(self, step_size: float = 1.0):
        """Initialization

        Parameters
        ----------
        step_size: float
            Standard deviation of normal distribution where the step in x, y coordinates is sampled
        """
        super().__init__()
        self.step_size = step_size

    def act(self, obs):
        """The base model executes a random action from a normal distribution
        Parameters
        ----------
        obs:
            Whatever observation from the environment class needed to choose the right action
        Returns
        -------
        d_pos: nd.array (2,)
            position variation to compute next position
        """
        d_pos = np.random.normal(scale=self.step_size, size=(2,))
        return d_pos


class LevyFlightAgent(RandomAgent):
    """Based on https://en.wikipedia.org/wiki/L%C3%A9vy_flight
    and https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levy_stable.html#scipy.stats.levy_stable
    Still experimental, need hyperparameter tuning and perhaps some momentum"""

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 1,
        loc: float = 1.0,
        scale: float = 0.8,
        step_size: float = 0.3,
        max_action_size: float = 50,
        max_step_size: float = 10,
    ):
        """Initializing levy flight agent
        From original documentation:
        The probability density above is defined in the “standardized” form. To shift and/or scale the distribution
        use the loc and scale parameters. Specifically, levy_stable.pdf(x, alpha, beta, loc, scale) is identically
        equivalent to levy_stable.pdf(y, alpha, beta) / scale with y = (x - loc) / scale. Note that shifting the
        location of a distribution does not make it a “noncentral” distribution; noncentral generalizations of some
        distributions are available in separate classes.

        Parameters
        ----------
        alpha, beta: float
            Levy flight distribution parameters
        loc: float
            bias of the standardized form
        scale: float
            scaling of the standardized form
        step_size: float
            direction scaling
        max_action_size: float
            maximum size of sampled step from levy distribution
        max_step_size: float
            maximum step size when multiplying max_action_size and step_size
        """
        super().__init__(step_size=step_size)
        self.levy = levy_stable(alpha, beta, loc=loc, scale=scale)
        self.alpha = alpha
        self.beta = beta
        self.max_action_size = max_action_size
        self.max_step_size = max_step_size
        self.action_buffer = []

    def _act(self, obs):
        """Auxiliary action method to compute

        Parameters
        ----------
        obs:
            Whatever observation from the environment class needed to choose the right action

        Returns
        -------
        d_pos: nd.array (2,)
            position variation to compute next position
        """
        # Pick direction
        direction = super().act(obs)
        # Normalize direction to step size
        direction = direction / np.sqrt(np.sum(direction**2)) * self.step_size
        # Sample step size
        r = np.clip(self.levy.rvs(), a_min=0, a_max=self.max_action_size)
        # Return step size from levy in a random direction
        d_pos = r * direction
        return d_pos

    def act(self, obs):
        """Sample levy flight steps. If steps are too large (action_size > max_step_size),
        it will divide it in several steps in the same direction.

        Parameters
        ----------
        obs:
            Whatever observation from the environment class needed to choose the right action

        Returns
        -------
        d_pos: nd.array (2,)
            position variation to compute next position
        """
        if len(self.action_buffer) > 0:
            action = self.action_buffer.pop()
            return action
        else:
            """
            Divide actions into multiple steps in the same direction
            (Need to refactor this feature)
            """
            action = self._act(obs)
            action_size = np.sqrt(np.sum(action**2))
            normalized_action = action / action_size
            if action_size > self.max_step_size:
                n_sub_steps = int(np.ceil(action_size / self.max_action_size))
                sub_actions = [normalized_action for i in range(n_sub_steps)]
                self.action_buffer += sub_actions
                return self.action_buffer.pop()
            else:
                return action
