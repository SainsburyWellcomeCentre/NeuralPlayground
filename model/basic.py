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
        
    def plot_rate(self,colormap='viridis',number_of_different_colors=30,):
        output_rates =
        linspace = np.linspace(-0.5 , 0.5, 51)
        X, Y = np.meshgrid(linspace, linspace)
        distance = np.sqrt(X*X + Y*Y)
        print(output_rates.shape)
        title = 'Plot of rate'
        plt.title(title, fontsize=12)
        cm = getattr(mpl.cm, colormap)
        cm.set_over('y', 1.0) # Set the color for values higher than maximum
        cm.set_bad('white', alpha=0.0)
        maximal_rate = int(np.ceil(np.amax(output_rates)))
        V = np.linspace(0, maximal_rate, number_of_different_colors)
        # Hack to avoid error in case of vanishing output rate at every position
        # If every entry in output_rates is 0, you define a norm and set
        # one of the elements to a small value (such that it looks like zero)
        if np.count_nonzero(output_rates) == 0:
            color_norm = mpl.colors.Normalize(0., 100.)
            output_rates[0][0] = 0.000001
            V = np.linspace(0, 2.0, number_of_different_colors)
            plt.contourf(X, Y, output_rates[...,0], V, norm=color_norm, cmap=cm, extend='max')
        else:
        a = output_rates[:, :, 28, 0]
        plt.contourf(X, Y, a, V, cmap=cm)
        plt.margins(0.01)
        plt.axis('off')
        ticks = np.linspace(0.0, maximal_rate, 2)
        plt.colorbar(format='%.2f', ticks=ticks)
        ax = plt.gca()
        self.set_axis_settings_for_contour_plots(ax)
        return
    

    
