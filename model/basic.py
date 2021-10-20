import sys
sys.path.append("../")
import numpy as np
import random
from model.core import NeuralResponseModel as NeurResponseModel
import matplotlib.pyplot as plt
import matplotlib as mpl
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
        self.rates_exc =[] ;

    def reset(self):
        self.global_steps = 0
        self.history = []
        self.wi = np.ones((self.Ni)) #what is the mu and why do we have the 1 and not2
        self.we = np.ones((self.Ne))
        self.rout = np.zeros((self.D,1))
 
    def act(self, observation):
        action=np.random.normal(scale=0.1, size=(2,))
        return action
 
    def get_rates_exc(self, x):
        self.rates_exc=np.zeros((self.Ne,1))
        for i in range(self.Ne):
            mean = np.random.normal(1)
            mean = mean # This mean should be calculated in a smarter way.
            rates_exc_i=self.alpha_exc*np.exp(((x-mean).T@(x-mean)/self.sigma_exc))
            self.rates_exc[i]= rates_exc_i
        return self.rates_exc
        
    def get_rates_inh(self, x):
        self.rates_inh= np.zeros((self.Ni,1))
        for i in range(self.Ni):
            mean = np.random.normal(1)
            mean = mean
            rates_inh_i= self.alpha_inh*np.exp(((x-mean).T@(x-mean)/self.sigma_inh))
            self.rates_inh[i]=rates_inh_i
        return self.rates_inh
        
    def output_rates(self, x):
        self.get_rates_exc_dup= self.get_rates_exc(x)
        self.get_rates_inh_dup= self.get_rates_inh(x)
        self.rout = (np.dot(self.we,self.get_rates_exc_dup)- np.dot(self.wi,self.get_rates_inh_dup))
        return self.rout

    def update(self, x):
        self.global_steps += 1
        self.rout=self.output_rates(x)
        self.we = self.we + self.etaexc * self.get_rates_exc_dup @ self.rout    # Weight update inh
        self.wi = self.wi +  self.etainh *  self.get_rates_inh_dup @ self.rout  # Weight update exc
        transition = {"wi": self.wi , "we": self.we, "self.rout": self.rout,}
        self.history.append(transition)
        
        return self.rout
        
    def plot_rate(self,room_width, room_depth, colormap='viridis',number_of_different_colors=30,):
             arena_limits = np.array([[-room_width/2, room_width/2],
             [-room_depth/2, room_depth/2]])
             # I would like to put as input the width of the env
             figsize=(8, 6)
             linspace_width = np.linspace(-room_width/2, room_width/2, 20)
             linspace = np.linspace(-room_depth/2, room_depth/2, 20)
             X, Y = np.meshgrid(linspace_width, linspace)
             a=np.zeros((len(X),len(X)))
             U= np.zeros((1,2))
             for i in range(len(X)):
                 for j in range(len(X.T)):
                     U=np.array([X[i,j],Y[i,j]])
                     a[i,j]=self.output_rates(U)
             title = 'Plot of rate'
             plt.title(title, fontsize=12)
             cm = getattr(mpl.cm, colormap)
             cm.set_over('y', 1.0) # Set the color for values higher than maximum
             cm.set_bad('white', alpha=0.0)
             maximal_rate = int(np.ceil(np.amax(a)))
             V = np.linspace(0, maximal_rate, number_of_different_colors)
             # Hack to avoid error in case of vanishing output rate at every position
             # If every entry in output_rates is 0, you define a norm and set
             # one of the elements to a small value (such that it looks like zero)
             plt.contourf(X, Y, a, V, cmap=cm)
             plt.margins(0.01)
             plt.axis('off')
             ticks = np.linspace(0.0, maximal_rate, 2)
             plt.colorbar(format='%.2f', ticks=ticks)
             ax = plt.gca()
             ax.plot([-room_width/2, room_width/2],
                     [-room_depth/2, -room_depth/2], "r", lw=2)
             ax.plot([-room_width/2, room_width/2],
                     [room_depth/2, room_depth/2], "r", lw=2)
             ax.plot([-room_width/2, -room_width/2],
                     [-room_depth/2, room_depth/2], "r", lw=2)
             ax.plot([room_width / 2, room_width / 2],
                     [-room_depth / 2, room_depth / 2], "r", lw=2)
             plt.show()
             return X
    

    
