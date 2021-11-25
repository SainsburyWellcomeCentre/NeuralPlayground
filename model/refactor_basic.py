import sys

sys.path.append("../")
import numpy as np
import random
from model.core import NeuralResponseModel as NeurResponseModel
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.stats import multivariate_normal

class ExcInhPlasticity(NeurResponseModel):

    def __init__(self, model_name="ExcitInhibitoryplastic", **mod_kwargs):
        super().__init__(model_name, **mod_kwargs)
        self.agent_step_size = mod_kwargs["agent_step_size"]
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
        self.room_width, self.room_depth = mod_kwargs["room_width"], mod_kwargs["room_depth"]
        self.D = 1  # Stimulus dimensions: punctate and contextual cues.
        self.reset()
        self.rates_exc = []

    def reset(self):
        self.global_steps = 0
        self.history = []
        self.wi = np.ones((self.Ni))*1.52  # what is the mu and why do we have the 1 and not2
        self.we = np.ones((self.Ne))
        self.sumwe = np.sum(self.we ** 2)
        self.xprev = [0, 0]

        self.inh_rates_functions = self.generate_tuning_curves(n_curves=self.Ni, cov_scale=self.sigma_inh)
        self.exc_rates_functions = self.generate_tuning_curves(n_curves=self.Ne, cov_scale=self.sigma_exc)

    def generate_tuning_curves(self, n_curves, cov_scale):
        width_limit = self.room_width / 2
        depth_limit = self.room_depth / 2
        mean1 = np.random.uniform(-width_limit, width_limit)
        mean2 = np.random.uniform(-depth_limit, depth_limit)
        cov = np.diag([width_limit*cov_scale, depth_limit*cov_scale])
        mean = [mean1, mean2]
        function_list = []
        for i in range(n_curves):
            function_list.append(multivariate_normal(mean, cov))
        return function_list

    def get_output_rates(self, x, y):
        r_out = 0
        pos = np.array([x, y])
        for i, exc_rate in enumerate(self.exc_rates_functions):
            r_out += exc_rate.pdf(pos)*self.we[i]
        for i, inh_rate in enumerate(self.inh_rates_functions):
            r_out += -inh_rate.pdf(pos)*self.wi[i]
        return np.abs(r_out)