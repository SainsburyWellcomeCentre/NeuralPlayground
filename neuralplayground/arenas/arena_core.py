import os
import pickle

import numpy as np
import pandas as pd
from deepdiff import DeepDiff
from gymnasium import Env, spaces


class Environment(Env):
    """Abstract parent environment class
    Methods
    ----------
    __init__(self, environment_name: str = "Environment", **env_kwargs):
        Initialize the class. env_kwargs arguments are specific for each of the child environments and
        described in their respective class
    make_observation(self):
        Returns the current state of the environment
    step(self, action):
        Runs the environment dynamics. Given some action, return observation, new state and reward.
    reset(self):
        Re-initialize state and global counters. Returns observation and re-setted state
    save_environment(self, save_path: str):
        Save current variables of the object to re-instantiate the environment later
    restore_environment(self, save_path: str):
        Restore environment saved using save_environment method
    get_trajectory_data(self):
        Returns interaction history

    Attributes
    ----------
    state: array
        Empty array for this abstract class
    history: list
        list containing transition history
    time_step_size: float
        time step in second traverse when calling step method
    metadata: dict
        dictionary with extra metadata that might be available in other classes
    env_kwags: dict
        Arguments given to the init method
    global_steps: int
        number of calls to step method, set to 0 when calling reset
    global_time: float
        time simulating environment through step method, then global_time = time_step_size * global_steps
    observation_space: gym.spaces
        specify the range of observations as in openai gym
    action_space: gym.spaces
        specify the range of actions as in openai gym
    """

    def __init__(self, environment_name: str = "Environment", **env_kwargs):
        """Initialisation of Environment class

        Parameters
        ----------
        environment_name: str
            environment name for the specific instantiation of the object
        env_kwargs: dict
            Define within each subclass for specific environments
            time_step_size: float
               Size of the time step in seconds
        """
        self.environment_name = environment_name
        if "time_step_size" in env_kwargs.keys():
            self.time_step_size = env_kwargs["time_step_size"]  # All environments should have this (second units)
        else:
            self.time_step_size = 1.0
        self.env_kwargs = env_kwargs  # Parameters of the environment
        self.metadata = {"env_kwargs": env_kwargs}  # Define within each subclass for specific environments
        self.state = np.array([])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(0,), dtype=np.float64)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(0,), dtype=np.float64)
        self.history = []
        # Initializing global counts
        self.global_steps = 0
        self.global_time = 0

    def make_observation(self):
        """Just take the state and returns an array of sensory information
        In more complex cases, the observation might be different from the internal state of the environment

        Returns
        -------
        self.state:
            Define within each subclass for specific environments
            Variable containing the state of the environment (eg. position in the environment)
        """
        return self.state

    def step(self, action=None):
        """Runs the environment dynamics. Increasing global counters.
        Given some action, return observation, new state and reward.

        Parameters
        ----------
        action:
            Abstract argument representing the action from the agent

        Returns
        -------
        observation:
            Define within each subclass for specific environments
            Any set of observation that is made by the agent in the environment (position, visual features,...)
        self.state:
            Define within each subclass for specific environments
            Variable containing the state of the environment (eg.position in the environment)
        reward: int
            The reward that the animal receives in this state transition
        """
        observation = self.make_observation()  # Build sensory info from current state
        reward = self.reward_function(action, self.state)
        self._increase_global_step()
        # state should be updated as well
        return observation, self.state, reward

    def _increase_global_step(self):
        self.global_steps += 1
        self.global_time += self.time_step_size

    def reset(self):
        """Re-initialize state. Returns observation and re-setted state

        Returns
        -------
        observation:
            Define within each subclass for specific environments.
            Any set of observation that is made by the agent in the environment (position, visual features,...)
        self.state:
            Define within each subclass for specific environments
            Variable containing the state of the environment (eg.position in the environment)
        """
        self.global_steps = 0
        self.global_time = 0
        observation = self.make_observation()
        return observation, self.state

    def save_environment(self, save_path: str):
        """Save current variables of the object to re-instantiate the environment later

        Parameters
        ----------
        save_path: str
            Path to save the environment
        """
        # pickle.dump(self.__dict__, open(os.path.join(save_path), "wb"), pickle.HIGHEST_PROTOCOL)
        pickle.dump(self, open(os.path.join(save_path), "wb"), pickle.HIGHEST_PROTOCOL)

    def restore_environment(self, save_path: str):
        """Restore environment saved using save_environment method

        Parameters
        ----------
        save_path: str
            Path to retrieve the environment
        """
        # self.__dict__ = pd.read_pickle(save_path)
        # TODO: for some reason, ruff has a problem with this: self = pd.read_pickle(save_path)
        pd.read_pickle(save_path)

    def __eq__(self, other):
        """Check if two environments are equal by comparing all of its attributes

        Parameters:
        ----------
        other: Environment
            Another instantiation of the environment
        return: bool
            True if self and other are the same exact environment
        """
        diff = DeepDiff(self.__dict__, other.__dict__)
        if len(diff) == 0:
            return True
        else:
            return False

    def get_trajectory_data(self):
        """Returns interaction history"""
        return self.history

    def reward_function(self, action, state):
        """Code reward curriculum here as a function of action and state
        and attributes of the environment if you want

        Parameters
        ----------
        action:
            same as step method argument
        state:
            same as state attribute of the class

        Returns
        -------
        reward: float
            reward given the parameters
        """
        reward = 0
        return reward


if __name__ == "__main__":
    env = Environment(environment_name="test_environment", time_step_size=0.5, one_kwarg_argument=10)
    print(env.__dict__)
    print(env.__dir__())
