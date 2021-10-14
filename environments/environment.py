from abc import ABC, abstractmethod
import numpy as np

class Environment(object):
    """Main environment class"""
    def __init__(self, environment_name="default_env", **env_kwargs):
        self.environment_name = environment_name
        self.env_kwargs = env_kwargs  # Variables to manipulate environment
        self.metadata = {"env_kwargs": env_kwargs}  # Define within each subclass for specific environments
        self.time_step_size = env_kwargs["time_step_size"]  # All environments should have this (second units)
        self.state = np.array([])
        self.history = []
        self.global_steps = 0

    def make_observation(self):
        """ Just take the state and returns and array of sensory information """
        return self.state

    def step(self, action):
        """ Given some action, return observation, new state and reward """
        observation = self.make_observation() # Build sensory info from current state
        reward = 0
        # state should be updated as well
        return observation, self.state, reward

    def reset(self):
        """ Re initialize state. Returns observation and re-setted state """
        observation = self.make_observation()
        return observation, self.state

    def save_environment(self, save_path):
        """ Save current state and information in general to re-instantiate the environment """
        pass

    def restore_environment(self, save_path):
        """ Restore saved environment """
        pass

    def get_trajectory_data(self):
        """ Return some sort of actions history """
        pass


if __name__ == "__main__":
    env = Environment(environment_name="test_environment", time_step_size=0.5, one_kwarg_argument=10)
    print(env.__dict__)
    print(env.__dir__())
