import sys
sys.path.append("../")
import numpy as np
import random
from model.core import NeuralResponseModel as NeurResponseModel
import matplotlib.pyplot as plt
import numpy as np

class ExcitInhibitoryplastic(NeurResponseModel):

    def __init__(self, model_name="ExcitInhibitoryplastic",**mod_kwargs):
        super().__init__(model_name,**mod_kwargs)
        self.metadata = {"mod_kwargs": mod_kwargs}
        self.etaexc = mod_kwargs["exc_eta"]  # Learning rate.
        self.etainh = mod_kwargs["inh_eta"]
        self.Ne=mod_kwargs["Ne"]
        self.Ni=mod_kwargs["Ni"]
        self.alpha_exc= mod_kwargs["alpha_exc"]
        self.sigma_exc= mod_kwargs["sigma_exc"]
        self.alpha_inh= mod_kwargs["alpha_inh"]
        self.sigma_inh= mod_kwargs["sigma_inh"]
        self.D = 1  # Stimulus dimensions: punctate and contextual cues.
        self.reset()

    def reset(self):
        self.global_steps = 0
        self.history = []
        self.wi = np.ones((self.Ni)) #what is the mu and why do we have the 1 and not2
        self.we = np.ones((self.Ne))
        rout = np.zeros((self.D,1))
 
    def act(self, observation):
        action=np.random.normal(scale=0.1, size=(2,))
        return action
 

    def update(self, x):
        self.global_steps += 1
        self.get_rates_exc= self.alpha_exc*np.exp((x.T@x/self.sigma_exc))
        self.get_rates_exc_dup = np.tile(self.get_rates_exc,(self.Ne,1))
        self.get_rates_inh= self.alpha_inh*np.exp((x.T@x/self.sigma_inh))
        self.get_rates_inh_dup = np.tile(self.get_rates_inh,(self.Ni,1))
        self.rout = (np.dot(self.we,self.get_rates_exc_dup)- np.dot(self.wi,self.get_rates_inh_dup))
        a=self.etaexc * self.get_rates_exc_dup @ self.rout
        print(a.shape)
        self.we = self.we + self.etaexc * self.get_rates_exc_dup @ self.rout    # Weight update inh
        self.wi = self.wi +  self.etainh *  self.get_rates_inh_dup @ self.rout  # Weight update exc
        transition = {"wi": self.wi , "we": self.we, "self.rout": self.rout,}
        self.history.append(transition)
        
        return self.rout
    

    
