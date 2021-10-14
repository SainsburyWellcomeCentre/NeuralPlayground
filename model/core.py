from abc import ABC, abstractmethod
import numpy as np
import math
import random

class NeuralResponseModel(object):
    """Abstract class for all models."""
    def __init__(self, model_name="default_model", **mod_kwargs):
        self.model_name = model_name
        self.mod_kwargs = mod_kwargs  # Variables to manipulate environment
        self.metadata = {"mod_kwargs": mod_kwargs}  # Define within each subclaspecificenvironments
        self.history = []
        self.global_steps = 0
 
    def reset(self):
        """Erase all memory from the model."""
        pass

    def neuralresponse(self):
        """Perform an action given current stimulus input
        observation dictionanary with:
        :param h: Head direction
        :param x: Position
        :param v: Velocity
        :param s: stimuli
        :param t: timestep within trial
        :param self: internal state
        return: neural response() (This might go to update self (state) state but this is unsure still)"""
        pass
        
    def act(self):
        """Perform an action given current stimulus input
        observation dictionanary with:
        :param h: Head direction
        :param x: Position
        :param v: Velocity
        :param s:  stimuli
        :param t: timestep within trial
        :param self: internal state
        return: update/move(x,v,h,s) vector which will go to update the environment observation
        """
        pass

if __name__ == "__main__":
    exc_eta_1=2e-6
    inh_eta_1= 8e-6
    model_name_1 = "model_example"
    sigma_exc_1= [0.05, 0.05]
    sigma_inh_1=[0.1, 0.1]
    Ne_1= 4900
    Ni_1= 1225
    alpha_inh_1=1
    alpha_exc_1=1
        
    agent=ExcitInhibitoryplastic(model_name= model_name_1,exc_eta=exc_eta_1,inh_eta=inh_eta_1, sigma_exc=sigma_exc_1, sigma_inh=sigma_inh_1, Ne=Ne_1, Ni= Ni_1, alpha_inh=alpha_inh_1,alpha_exc=alpha_exc_1)
    print(env.__dict__)
    print(env.__dir__())



