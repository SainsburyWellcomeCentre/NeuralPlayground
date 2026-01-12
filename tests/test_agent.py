import numpy as np
import pytest
from tqdm import tqdm

from neuralplayground.agents import AgentCore, Stachenfeld2018, Weber2018
from neuralplayground.agents.whittington_2020 import Whittington2020
from neuralplayground.agents.whittington_2020_extras import whittington_2020_parameters as parameters
from neuralplayground.arenas import BasicSargolini2006, BatchEnvironment, DiscreteObjectEnvironment
from neuralplayground.experiments import Sargolini2006Data


@pytest.fixture
def get_environment():
    env = BasicSargolini2006()
    return [
        env,
    ]


class Testmodelcore(object):
    @pytest.fixture
    def init_model(self, get_environment):
        agent = AgentCore()

        return [
            agent,
        ]

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

        agent = Weber2018(
            model_name=model_name,
            exc_eta=exc_eta,
            inh_eta=inh_eta,
            sigma_exc=sigma_exc,
            sigma_inh=sigma_inh,
            Ne=Ne,
            Ni=Ni,
            agent_step_size=agent_step_size,
            ro=1,
            Nef=Nef,
            Nif=Nif,
            room_width=env.room_width,
            room_depth=env.room_depth,
            alpha_i=alpha_i,
            alpha_e=alpha_e,
            we_init=we_init,
            wi_init=wi_init,
        )
        return [
            agent,
        ]

    def test_init_model(self, init_model):
        assert isinstance(init_model[0], Weber2018)

    def test_plot_rates(self, init_model):
        init_model[0].plot_rate_map()

    def test_agent_interaction(self, init_model, get_environment):
        env = get_environment[0]
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
        discount = 0.9
        threshold = 1e-6
        lr_td = 1e-2
        t_episode = 50
        n_episode = 50
        state_density = 1 / agent_step_size
        env = get_environment[0]
        twoDvalue = True
        agent = Stachenfeld2018(
            discount=discount,
            t_episode=t_episode,
            n_episode=n_episode,
            threshold=threshold,
            lr_td=lr_td,
            room_width=env.room_width,
            room_depth=env.room_depth,
            state_density=state_density,
            twoD=twoDvalue,
        )
        return [
            agent,
        ]

    def test_agent_interaction(self, init_model, get_environment):
        env = get_environment[0]
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
        sr = init_model[0].successor_rep_solution()  # Choose your type of Update
        init_model[0].plot_rate_map(sr, eigen_vectors=(0,), save_path=None)
        init_model[0].plot_rate_map(sr, eigen_vectors=(0, 1), save_path=None)

    def test_plot_sr_td(self, init_model):
        sr_td = init_model[0].update_successor_rep_td_full()  # Choose your type of Update
        init_model[0].plot_rate_map(sr_td, eigen_vectors=(0,), save_path=None)

    def test_plot_sr_sum(self, init_model):
        sr_sum = init_model[0].successor_rep_sum()
        init_model[0].plot_rate_map(sr_sum, eigen_vectors=(0,), save_path=None)


class TestWhittington2020(Testmodelcore):
    @pytest.fixture
    def init_model(self, get_environment):
        mod_name = "Whittington2020_test"
        pars = parameters.parameters()
        params = pars.copy()
        batch_size = 16

        agent = Whittington2020(
            model_name=mod_name,
            params=params,
            batch_size=batch_size,
            room_widths=[10] * batch_size,
            room_depths=[10] * batch_size,
            state_densities=[1] * batch_size,
            use_behavioural_data=False,
        )
        return [
            agent,
        ]

    def test_agent_interaction(self):
        agent_params = parameters.parameters()
        discrete_env_params = {
            "environment_name": "DiscreteObject",
            "state_density": 1,
            "n_objects": 45,
            "agent_step_size": 1,
            "use_behavioural_data": False,
            "data_path": None,
            "experiment_class": Sargolini2006Data,
        }
        env = BatchEnvironment(
            environment_name="BatchEnvironment",
            batch_size=16,
            arena_x_limits=[
                [-5, 5],
                [-4, 4],
                [-5, 5],
                [-6, 6],
                [-4, 4],
                [-5, 5],
                [-6, 6],
                [-5, 5],
                [-4, 4],
                [-5, 5],
                [-6, 6],
                [-5, 5],
                [-4, 4],
                [-5, 5],
                [-6, 6],
                [-5, 5],
            ],
            arena_y_limits=[
                [-5, 5],
                [-4, 4],
                [-5, 5],
                [-6, 6],
                [-4, 4],
                [-5, 5],
                [-6, 6],
                [-5, 5],
                [-4, 4],
                [-5, 5],
                [-6, 6],
                [-5, 5],
                [-4, 4],
                [-5, 5],
                [-6, 6],
                [-5, 5],
            ],
            env_class=DiscreteObjectEnvironment,
            arg_env_params=discrete_env_params,
        )
        room_widths = [10, 8, 10, 12, 8, 10, 12, 10, 8, 10, 12, 10, 8, 10, 12, 10]
        room_depths = [10, 8, 10, 12, 8, 10, 12, 10, 8, 10, 12, 10, 8, 10, 12, 10]
        agent = Whittington2020(
            model_name="Whittington2020_test",
            params=agent_params.copy(),
            batch_size=16,
            room_widths=room_widths,
            room_depths=room_depths,
            state_densities=[1] * 16,
            use_behavioural_data=False,
        )
        n_steps = 1
        obs, state = env.reset()
        for i in tqdm(range(n_steps)):
            while agent.n_walk < agent.pars["n_rollout"]:
                actions = agent.batch_act(obs)
                obs, state, reward = env.step(actions, normalize_step=True)

    def test_agent_update(self):
        agent_params = parameters.parameters()
        arena_x_limits = [
            [-5, 5],
            [-4, 4],
            [-5, 5],
            [-6, 6],
            [-4, 4],
            [-5, 5],
            [-6, 6],
            [-5, 5],
            [-4, 4],
            [-5, 5],
            [-6, 6],
            [-5, 5],
            [-4, 4],
            [-5, 5],
            [-6, 6],
            [-5, 5],
        ]
        arena_y_limits = [
            [-5, 5],
            [-4, 4],
            [-5, 5],
            [-6, 6],
            [-4, 4],
            [-5, 5],
            [-6, 6],
            [-5, 5],
            [-4, 4],
            [-5, 5],
            [-6, 6],
            [-5, 5],
            [-4, 4],
            [-5, 5],
            [-6, 6],
            [-5, 5],
        ]
        room_widths = [10, 8, 10, 12, 8, 10, 12, 10, 8, 10, 12, 10, 8, 10, 12, 10]
        room_depths = [10, 8, 10, 12, 8, 10, 12, 10, 8, 10, 12, 10, 8, 10, 12, 10]
        discrete_env_params = {
            "environment_name": "DiscreteObject",
            "state_density": 1,
            "n_objects": 45,
            "agent_step_size": 1,
            "use_behavioural_data": False,
            "data_path": None,
            "experiment_class": Sargolini2006Data,
        }
        env = BatchEnvironment(
            environment_name="BatchEnvironment",
            batch_size=16,
            arena_x_limits=arena_x_limits,
            arena_y_limits=arena_y_limits,
            env_class=DiscreteObjectEnvironment,
            arg_env_params=discrete_env_params,
        )
        agent = Whittington2020(
            model_name="Whittington2020_test",
            params=agent_params.copy(),
            batch_size=16,
            room_widths=room_widths,
            room_depths=room_depths,
            state_densities=[1] * 16,
            use_behavioural_data=False,
        )
        n_steps = 1
        obs, state = env.reset(random_state=True, custom_state=None)
        for i in tqdm(range(n_steps)):
            while agent.n_walk < agent.pars["n_rollout"]:
                actions = agent.batch_act(obs)
                obs, state, reward = env.step(actions, normalize_step=True)
            agent.update()

    def test_plot_rates(self, init_model):
        init_model[0].plot_rate_map(rate_map_type="g")

    def test_init_model(self, init_model):
        assert isinstance(init_model[0], Whittington2020)
