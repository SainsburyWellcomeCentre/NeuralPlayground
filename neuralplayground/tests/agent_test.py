import sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from ..arenas import BasicSargolini2006
from ..arenas import BatchEnvironment
from ..arenas import DiscreteObjectEnvironment
from ..agents import Weber2018
from ..agents import Whittington2020
import neuralplayground.agents.whittington_2020_extras.whittington_2020_parameters as parameters
from ..agents import Stachenfeld2018
from ..agents import AgentCore
import neuralplayground.agents.whittington_2020_extras.whittington_2020_parameters as parameters
import pytest


@pytest.fixture
def get_environment():
    env = BasicSargolini2006()
    return [env, ]


@pytest.fixture
def get_batch_environment():
    pars_orig = parameters.parameters()
    params = pars_orig.copy()
    state_density = 1
    arena_x_limits = [[-5, 5], [-4, 4], [-5, 5], [-6, 6], [-4, 4], [-5, 5], [-6, 6], [-5, 5], [-4, 4], [-5, 5], [-6, 6],
                      [-5, 5], [-4, 4], [-5, 5], [-6, 6], [-5, 5]]
    arena_y_limits = [[-5, 5], [-4, 4], [-5, 5], [-6, 6], [-4, 4], [-5, 5], [-6, 6], [-5, 5], [-4, 4], [-5, 5], [-6, 6],
                      [-5, 5], [-4, 4], [-5, 5], [-6, 6], [-5, 5]]
    env_name = "env_example"
    mod_name = "TorchTEMTest"
    time_step_size = 1
    agent_step_size = 1 / state_density
    n_objects = 45
    batch_size = 16
    env_class = DiscreteObjectEnvironment
    env = BatchEnvironment(environment_name=env_name,
                           env_class=env_class,
                           batch_size=batch_size,
                           n_objects=n_objects,
                           arena_x_limits=arena_x_limits,
                           arena_y_limits=arena_y_limits,
                           time_step_size=time_step_size,
                           agent_step_size=agent_step_size,
                           state_density=state_density)
    return [env, ]


class Testmodelcore(object):

    @pytest.fixture
    def init_model(self, get_environment):
        agent = AgentCore()

        return [agent, ]

    def test_init_model(self, init_model):
        assert isinstance(init_model[0], AgentCore)


class TestWeber2018(Testmodelcore):

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

        agent = Weber2018(model_name=model_name, exc_eta=exc_eta, inh_eta=inh_eta, sigma_exc=sigma_exc,
                          sigma_inh=sigma_inh, Ne=Ne, Ni=Ni, agent_step_size=agent_step_size, ro=1,
                          Nef=Nef, Nif=Nif, room_width=env.room_width, room_depth=env.room_depth,
                          alpha_i=alpha_i, alpha_e=alpha_e, we_init=we_init, wi_init=wi_init)
        return [agent, ]

    def test_init_model(self, init_model):
        assert isinstance(init_model[0], Weber2018)

    def test_plot_rates(self, init_model):
        init_model[0].plot_rates()

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


class TestStachenfeld2018(Testmodelcore):

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
        agent = Stachenfeld2018(discount=discount, t_episode=t_episode, n_episode=n_episode, threshold=threshold,
                                lr_td=lr_td,
                                room_width=env.room_width, room_depth=env.room_depth, state_density=state_density,
                                twoD=twoDvalue)
        return [agent, ]

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

    def test_init_model(self, init_model):
        assert isinstance(init_model[0], Stachenfeld2018)

    def test_plot_sr_ground_truth(self, init_model):
        sr = init_model[0].update_successor_rep()  # Choose your type of Update
        init_model[0].plot_eigen(sr, eigen=(0,), save_path=None)
        init_model[0].plot_eigen(sr, eigen=(0, 1), save_path=None)

    def test_plot_sr_td(self, init_model):
        sr_td = init_model[0].update_successor_rep_td_full()  # Choose your type of Update
        init_model[0].plot_eigen(sr_td, eigen=(0,), save_path=None)

    def test_plot_sr_sum(self, init_model):
        sr_sum = init_model[0].successor_rep_sum()
        init_model[0].plot_eigen(sr_sum, eigen=(0,), save_path=None)


class TestWhittington2020(Testmodelcore):

    @pytest.fixture
    def init_model(self, get_batch_environment):
        pars_orig = parameters.parameters()
        params = pars_orig.copy()
        mod_name = "TorchTEMTest"
        batch_size = 16
        env = get_batch_environment[0]
        # Init environment
        agent = Whittington2020(model_name=mod_name,
                                params=params,
                                batch_size=batch_size,
                                room_widths=env.room_widths,
                                room_depths=env.room_depths,
                                state_densities=env.state_densities)
        return [agent, ]

    def test_agent_interaction(self, init_model, get_batch_environment):
        pars_orig = parameters.parameters()
        params = pars_orig.copy()
        env = get_batch_environment[0]
        observation, state = env.reset(random_state=False, custom_state=None)
        for i in range(1):
            while init_model[0].n_walk < params['n_rollout']:
                actions = init_model[0].batch_act(observation)
                observation, state = env.step(actions, normalize_step=True)
            init_model[0].update()