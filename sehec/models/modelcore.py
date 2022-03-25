"""
Base class for models that can interact with environments in this repo
Any EHC model should inherit this class in order to interact with environments and compare against experimental results
We expect to make profound changes in this module as we add more EHC model to the repo
"""
import numpy as np


class NeuralResponseModel(object):
    """ Abstract class for all EHC models

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
        self.metadata = {"mod_kwargs": mod_kwargs}
        self.obs_history = []
        self.global_steps = 0
 
    def reset(self):
        """Erase all memory from the model, initialize all relevant parameters and build from scratch """
        pass

    def neural_response(self):
        """ Function that returns some representation that will be compared against real experimental data """
        pass
        
    def act(self, obs):
        """
        The base model executes a random action from a normal distribution
        Parameters
        ----------
        obs
            Whatever observation from the environment class needed to choose the right action
        Returns
        -------
        action: float
            action value which in this case is random number draw from a Gaussian
        """
        self.obs_history.append(obs)
        if len(self.obs_history) >= 1000:  # reset every 1000
            self.obs_history = [obs, ]
        action = np.random.normal(scale=0.1, size=(2,))
        return action
