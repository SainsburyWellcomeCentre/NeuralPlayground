import numpy as np
import matplotlib.pyplot as plt
from ..envs.arenas.simple2d import Simple2D, Sargolini2006, BasicSargolini2006, Hafting2008
import pytest


class RandomAgent(object):

    def act(self, observation):
        return np.random.normal(scale=0.1, size=(2,))


class TestSimple2D(object):

    @pytest.fixture
    def init_env(self):
        room_width = 15
        room_depth = 7
        env_name = "env_example"
        time_step_size = 0.1
        agent_step_size = 0.5

        # Init environment
        env = Simple2D(environment_name=env_name,
                       room_width=room_width,
                       room_depth=room_depth,
                       time_step_size=time_step_size,
                       agent_step_size=agent_step_size)
        return [env, ]

    def test_init_env(self, init_env):
        assert isinstance(init_env[0], Simple2D)

    def test_agent_interaction(self, init_env):
        n_steps = 1
        agent = RandomAgent()
        # Initialize environment
        obs, state = init_env[0].reset()
        for i in range(n_steps):
            # Observe to choose an action
            action = agent.act(obs)
            # Run environment for given action
            obs, state, reward = init_env[0].step(action)
        init_env[0].plot_trajectory()


class TestBasicSargolini2006(TestSimple2D):
    @pytest.fixture
    def init_env(self):
        data_path = "../envs/experiments/Sargolini2006/"
        env = BasicSargolini2006(data_path=data_path,
                                 time_step_size=0.1,
                                 agent_step_size=None)
        return [env, ]

    def test_init_env(self, init_env):
        assert isinstance(init_env[0], BasicSargolini2006)


class TestSargolini2006(TestSimple2D):
    @pytest.fixture
    def init_env(self):
        data_path = "../envs/experiments/Sargolini2006/raw_data_sample/"

        session = {"rat_id": "11016", "sess": "31010502"}

        env = Sargolini2006(data_path=data_path,
                            verbose=True,
                            session=session,
                            time_step_size=None,
                            agent_step_size=None)
        return [env, ]

    def test_init_env(self, init_env):
        assert isinstance(init_env[0], Sargolini2006)

    def test_session_plot(self, init_env):
        init_env[0].data.plot_session()


class TestHafting2008(TestSimple2D):
    @pytest.fixture
    def init_env(self):
        data_path = "../envs/experiments/Hafting2008/"
        session = {"rat_id": "11015", "sess": "13120410", "cell_id": "t5c1"}

        env = Hafting2008(data_path=data_path,
                          verbose=True,
                          session=session,
                          time_step_size=None, agent_step_size=None)
        return [env, ]

    def test_init_env(self, init_env):
        assert isinstance(init_env[0], Hafting2008)