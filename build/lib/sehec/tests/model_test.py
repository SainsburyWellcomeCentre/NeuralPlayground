import sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from ..envs.arenas.simple2d import Simple2D, Sargolini2006, Hafting2008,BasicSargolini2006
from ..models.weber_and_sprekeler import ExcInhPlasticity
import pytest


@pytest.fixture
def get_environment():
    data_path = "../envs/experiments/Sargolini2006/"
    env = BasicSargolini2006(data_path=data_path,
                             time_step_size=0.1,
                             agent_step_size=None)
    return [env, ]


class TestExcInhPlasticity(object):

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

    def test_agent_interaction(self, init_model, get_environment):
        env = get_environment[0]
        plot_every = 0
        total_iters = 0
        n_steps = 1

        obs, state = env.reset()
        # for i in tqdm(range(env.total_number_of_steps)):
        for i in tqdm(range(n_steps)):
            # Observe to choose an action
            obs = obs[:2]
            action = init_model[0].act(obs)
            # rate = agent.update()
            init_model[0].update()
            # Run environment for given action
            obs, state, reward = env.step(action)
            total_iters += 1

    def test_plot_rates(self, init_model):
        init_model[0].plot_rates()