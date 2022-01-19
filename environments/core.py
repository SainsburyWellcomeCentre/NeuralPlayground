from abc import ABC, abstractmethod
import numpy as np


class Environment(object):
    """Main environment class"""
    def __init__(self, environment_name="default_env", **env_kwargs):
        """ Initialisation of the class

        Parameters
        ----------
        environment_name: str
            Environment name
        env_kwargs
            Define within each subclass for specific environments
            time_step_size: float
               Size of the time step relative to real time
        """
        self.environment_name = environment_name
        self.env_kwargs = env_kwargs  # Variables to manipulate environment
        self.metadata = {"env_kwargs": env_kwargs}  # Define within each subclass for specific environments
        self.time_step_size = env_kwargs["time_step_size"]  # All environments should have this (second units)
        self.state = np.array([])
        self.history = []
        self.global_steps = 0
        self.global_time = 0

    def make_observation(self):
        """  Just take the state and returns and array of sensory information

        Returns
        -------
        self.state:
            Define within each subclass for specific environments
            Variable containing the state of the agent (eg.position in the environment)
        """
        return self.state

    def step(self):
        """ Given some action, return observation, new state and reward

        Returns
        -------
        reward: int
            The reward that the animal recieves in this state
        observation:
            Define within each subclass for specific environments
            Any set of observation that is made by the agent in the environment (position, visual features,...)
        self.state:
            Define within each subclass for specific environments
            Variable containing the state of the agent (eg.position in the environment)
        """
        observation = self.make_observation() # Build sensory info from current state
        reward = 0
        # state should be updated as well
        return observation, self.state, reward

    def reset(self):
        """ Re initialize state. Returns observation and re-setted state

        Returns
        -------
        observation:
            Define within each subclass for specific environments.
            Any set of observation that is made by the agent in the environment (position, visual features,...)
        self.state:
            Define within each subclass for specific environments
            Variable containing the state of the agent (eg.position in the environment)
        """
        observation = self.make_observation()
        return observation, self.state

    def save_environment(self, save_path):
        """ Save current state and information in general to re-instantiate the environment

        Parameters
        ----------
        save_path: str
            Path to save the environment
        """
        pass

    def restore_environment(self, save_path):
        """ Restore saved environment

        Parameters
        ----------
        save_path: str
            Path to retrieve the environment
        """
        pass

    def get_trajectory_data(self):
        """ Return some sort of actions history """
        pass


if __name__ == "__main__":
    env = Environment(environment_name="test_environment", time_step_size=0.5, one_kwarg_argument=10)
    print(env.__dict__)
    print(env.__dir__())
