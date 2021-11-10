import sys

sys.path.append("../")
import numpy as np
import random
from model.core import NeuralResponseModel as NeurResponseModel
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math


class ExcitInhibitoryplastic(NeurResponseModel):

    def __init__(self, model_name="ExcitInhibitoryplastic", **mod_kwargs):
        super().__init__(model_name, **mod_kwargs)
        self.agent_step_size= mod_kwargs["step"]
        self.metadata = {"mod_kwargs": mod_kwargs}
        self.etaexc = mod_kwargs["exc_eta"]  # Learning rate.
        self.etainh = mod_kwargs["inh_eta"]
        self.Ne = mod_kwargs["Ne"]
        self.Ni = mod_kwargs["Ni"]
        self.alpha_exc = mod_kwargs["alpha_exc"]
        self.sigma_exc = mod_kwargs["sigma_exc"]
        self.sigma_exc_1 = mod_kwargs["sigma_exc_1"]
        self.alpha_inh = mod_kwargs["alpha_inh"]
        self.sigma_inh = mod_kwargs["sigma_inh"]
        self.sigma_inh_1 = mod_kwargs["sigma_inh_1"]
        self.D = 1  # Stimulus dimensions: punctate and contextual cues.
        self.reset()
        self.rates_exc = [];

    def reset(self):
        self.global_steps = 0
        self.history = []
        self.wi = np.ones((self.Ni))  # what is the mu and why do we have the 1 and not2
        self.we = np.ones((self.Ne))
        self.sumwe = np.sum(self.we ** 2)
        self.xprev = [0, 0]

    def act(self, observation):
        action = np.random.normal(scale=0.1, size=(2,))
        return action

    def get_rates_exc(self, x, room_width, room_depth):
        rates_exc_save = np.zeros((self.Ne, 1))
        for i in range(self.Ne):
            A = round((room_width/2)*1000)
            B = round((room_depth / 2) * 1000)
            mean1 = np.random.randint(-A,A)*0.001
            mean2 = np.random.randint(-B, B) * 0.001
            mean=[mean1, mean2]
            mean = [-0.75, -0.75]
            mean=np.array(mean)
            l_exc=(x-mean)
            sig_exc=[1/(self.sigma_exc_1**2),1/(self.sigma_exc**2)]
            sigma=np.diag(np.array(sig_exc))
            rates_exc_i = self.alpha_exc * np.exp(l_exc.T @ sigma@ (l_exc))
            rates_exc_save[i] = rates_exc_i
        return rates_exc_save

    def get_rates_inh(self, x, room_width, room_depth, ):
        rates_inh_save = np.zeros((self.Ni, 1))
        for i in range(self.Ni):
            A = round((room_width / 2) * 1000)
            B = round((room_depth / 2) * 1000)
            mean1 = np.random.randint(-A, A) * 0.001
            mean2 = np.random.randint(-B, B) * 0.001
            mean = [mean1, mean2]
            mean = [0.75, 0.75]
            mean = np.array(mean)
            l_inh = (x - mean)
            sig_inh = [1/(self.sigma_inh_1**2), 1/(self.sigma_inh_1**2)]
            sigma_inh = np.diag(np.array(sig_inh))
            rates_inh_i = self.alpha_inh * np.exp(l_inh.T @ sigma_inh @ (l_inh))
            rates_inh_save[i] = rates_inh_i
        return rates_inh_save

    def get_rates_exc_select(self, x, room_width, Number_to_select, room_depth,):
        rates_exc_save = np.zeros(( Number_to_select, 1))
        for i in range( Number_to_select):
            A = round((room_width / 2) * 1000)
            B = round((room_depth / 2) * 1000)
            mean1 = np.random.randint(-A, A) * 0.001
            mean2 = np.random.randint(-B, B) * 0.001
            mean=[mean1, mean2]
            mean = [-0.75, -0.75]
            mean=np.array(mean)
            l_exc=(x-mean)
            sig_exc=[1/(self.sigma_exc_1**2),1/(self.sigma_exc**2)]
            sigma=np.diag(np.array(sig_exc))
            rates_exc_i = self.alpha_exc * np.exp(l_exc.T @ sigma@ (l_exc))
            rates_exc_save[i] = rates_exc_i
        return rates_exc_save

    def get_rates_inh_select(self, x, room_width, Number_to_select, room_depth, ):
        rates_inh_save = np.zeros(( Number_to_select, 1))
        for i in range( Number_to_select):
            A = round((room_width / 2) * 1000)
            B = round((room_depth / 2) * 1000)
            mean1 = np.random.randint(-A, A) * 0.001
            mean2 = np.random.randint(-B, B) * 0.001
            mean = [mean1, mean2]
            mean= [0.75, 0.75]
            mean = np.array(mean)
            l_inh = (x - mean)
            sig_inh = [1/(self.sigma_inh_1**2), 1/(self.sigma_inh**2)]
            sigma_inh = np.diag(np.array(sig_inh))
            rates_inh_i = self.alpha_inh * np.exp(l_inh.T @ sigma_inh @ (l_inh))
            rates_inh_save[i] = rates_inh_i
        return rates_inh_save


    def output_rates(self, x, room_width, room_depth):
        self.get_rates_exc_dup = self.get_rates_exc(x, room_width, room_depth)
        self.get_rates_inh_dup = self.get_rates_inh(x, room_width, room_depth)
        rout = (np.dot(self.we, self.get_rates_exc_dup) - np.dot(self.wi, self.get_rates_inh_dup))
        if rout < 0:
            rout=[0]
            rout=np.array(rout)
        print(rout)
        return rout

    def output_rates_select(self,x,room_width, room_depth):
        Number_to_select_i=self.Ni
        Number_to_select_e = self.Ne
        self.get_rates_exc_dup_select = self.get_rates_exc_select(x, room_width, Number_to_select_e, room_depth)
        self.get_rates_inh_dup_select = self.get_rates_inh_select(x, room_width, Number_to_select_i,  room_depth)
        # I do this this way because I select the first one in the above get_rates_inh_dup_select
        wi_select = self.wi[0:Number_to_select_i]
        we_select=self.we[0:Number_to_select_e]
        rout = (np.dot(we_select, self.get_rates_exc_dup_select) - np.dot(wi_select, self.get_rates_inh_dup_select))
        if rout < 0:
            rout = [0]
            rout = np.array(rout)
        print(rout)
        return rout

    def update(self, action, room_width, room_depth):
        action = action / np.linalg.norm(action)
        x= self.xprev + self.agent_step_size * action
        x = np.array([np.clip(x[0], a_min=-room_width / 2, a_max=room_width / 2),
                              np.clip(x[1], a_min=-room_depth / 2, a_max=room_depth / 2)])
        self.global_steps += 1
        rout = self.output_rates(x, room_width, room_depth)
        self.we = self.we + self.etaexc * self.get_rates_exc_dup @ rout  # Weight update inh
        rho=[1]
        rho= np.array(rho)
        self.wi = self.wi + (self.etainh * self.get_rates_inh_dup @ rout - self.etainh * self.get_rates_inh_dup@rho )  # Weight update exc
        l= (self.sumwe/sum((self.we))**2)
        self.we=self.we*l
        transition = {"wi": self.wi, "we": self.we, "self.rout": rout, }
        self.history.append(transition)
        self.xprev=x
        return rout

    def plot_rate(self, room_width, room_depth, colormap='viridis', number_of_different_colors=30, ):
        arena_limits = np.array([[-room_width / 2, room_width / 2],
                                 [-room_depth / 2, room_depth / 2]])
        # I would like to put as input the width of the env
        figsize = (8, 6)
        linspace_width = np.linspace((-room_width / 2), (room_width / 2), 30)
        linspace = np.linspace((-room_depth / 2), (room_depth / 2), 30)
        X, Y = np.meshgrid(linspace_width, linspace)
        a = np.zeros((len(X), len(X)))
        U = np.zeros((1, 2))
        for i in range(len(X)):
            for j in range(len(Y)):
                U = np.array([X[i, j], Y[i, j]])
                a[i,j] = self.output_rates_select(U, room_width, room_depth)
        title = 'Plot of rate'
        plt.title(title, fontsize=12)
        cm = getattr(mpl.cm, colormap)
        print(a)
        cm.set_over('y', 1.0)  # Set the color for values higher than maximum
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
        ax.plot([-room_width / 2, room_width / 2],
                [-room_depth / 2, -room_depth / 2], "r", lw=2)
        ax.plot([-room_width / 2, room_width / 2],
                [room_depth / 2, room_depth / 2], "r", lw=2)
        ax.plot([-room_width / 2, -room_width / 2],
                [-room_depth / 2, room_depth / 2], "r", lw=2)
        ax.plot([room_width / 2, room_width / 2],
                [-room_depth / 2, room_depth / 2], "r", lw=2)
        plt.show()
        return X

    
