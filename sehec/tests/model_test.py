import sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from ..envs.arenas.simple2d import Simple2D, Sargolini2006, Hafting2008,BasicSargolini2006
from ..models.weber_and_sprekeler import ExcInhPlasticity
from ..models.SRKim import SR
from ..models.modelcore import NeuralResponseModel
import pytest


@pytest.fixture
def get_environment():
    data_path = "../envs/experiments/Sargolini2006/"
    env = BasicSargolini2006(data_path=data_path,
                             time_step_size=0.1,
                             agent_step_size=None)
    return [env, ]

class Testmodelcore(object):

    @pytest.fixture
    def init_model(self, get_environment):
        agent= NeuralResponseModel()

        return [agent, ]

    def test_init_model(self, init_model):
        assert isinstance(init_model[0],NeuralResponseModel)

    def test_agent_interaction(self, init_model, get_environment):
        env = get_environment[0]
        plot_every = 0
        total_iters = 0
        n_steps = 1
        obs, state = env.reset()
        obs = obs[:2]
        # for i in tqdm(range(env.total_number_of_steps)):
        for i in tqdm(range(n_steps)):
            # Observe to choose an action
            action = init_model[0].act(obs)
            init_model[0].update()
            # rate = agent.update()
            # Run environment for given action
            obs, state, reward = env.step(action)
            obs = obs[:2]
            total_iters += 1

class TestExcInhPlasticity(Testmodelcore):

    @pytest.fixture
    def init_model(self, get_environment):
        exc_eta = 2e-4
        inh_eta = 8e-4
        model_name = "model_example"
        sigma_exc = np.array([0.05, 0.05])
        sigma_inh = np.array([0.1, 0.1])
        Ne = 10
        Ni = 10
        Nef = 1
        Nif = 1
        alpha_i = 1
        alpha_e = 1
        we_init = 1.0
        wi_init = 1.5
        agent_step_size = 0.1
        env = get_environment[0]

        agent = ExcInhPlasticity(model_name=model_name, exc_eta=exc_eta, inh_eta=inh_eta, sigma_exc=sigma_exc,
                                 sigma_inh=sigma_inh, Ne=Ne, Ni=Ni, agent_step_size=agent_step_size, ro=1,
                                 Nef=Nef, Nif=Nif, room_width=env.room_width, room_depth=env.room_depth,
                                 alpha_i=alpha_i, alpha_e=alpha_e, we_init=we_init, wi_init=wi_init)
        return [agent, ]

    def test_init_model(self, init_model):
        assert isinstance(init_model[0], ExcInhPlasticity)


    def test_plot_rates(self, init_model):
        init_model[0].plot_rates()

class TestSR(Testmodelcore):

    @pytest.fixture
    def init_model(self, get_environment):
        agent_step_size = 10
        discount = .9
        threshold = 1e-6
        lr_td = 1e-2
        t_episode = 50
        n_episode = 50
        state_density = (1 / agent_step_size)
        env = get_environment[0]

        twoDvalue = True
        agent = SR(discount=discount, t_episode=t_episode, n_episode=n_episode, threshold=threshold, lr_td=lr_td,
                   room_width=env.room_width, room_depth=env.room_depth, state_density=state_density, twoD=twoDvalue)
        return [agent, ]

    def test_init_model(self, init_model):
        assert isinstance(init_model[0], SR)

    def test_plot_sr_ground_truth(self, init_model):
        sr = init_model[0].update_successor_rep()  # Choose your type of Update
        init_model[0].plot_eigen(sr, save_path=None)

    def test_plot_sr_td(self, init_model):
        sr_td = init_model[0].update_successor_rep_td_full()  # Choose your type of Update
        init_model[0].plot_eigen(sr_td, save_path=None)

    def test_plot_sr_sum(self, init_model):
        sr_sum = init_model[0].successor_rep_sum()
        init_model[0].plot_eigen(sr_sum, save_path=None)