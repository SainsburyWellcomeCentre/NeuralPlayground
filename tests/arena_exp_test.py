import numpy as np
import pytest

import neuralplayground.agents.whittington_2020_extras.whittington_2020_parameters as parameters
from neuralplayground.agents import RandomAgent
from neuralplayground.agents.whittington_2020 import Whittington2020
from neuralplayground.arenas import (
    BatchEnvironment,
    ConnectedRooms,
    DiscreteObjectEnvironment,
    Hafting2008,
    MergingRoom,
    Sargolini2006,
    Simple2D,
    Wernle2018,
)
from neuralplayground.experiments import Sargolini2006Data


class TestSimple2D(object):
    @pytest.fixture
    def init_env(self):
        room_width = 15
        room_depth = 7
        env_name = "env_example"
        time_step_size = 0.1
        agent_step_size = 0.5

        # Init environment
        env = Simple2D(
            environment_name=env_name,
            arena_x_limits=np.array([-room_width / 2, room_width / 2]),
            arena_y_limits=np.array([-room_depth / 2, room_depth / 2]),
            time_step_size=time_step_size,
            agent_step_size=agent_step_size,
        )
        return [
            env,
        ]

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
            init_env[0].render()
        init_env[0].plot_trajectory()


class TestSargolini2006(TestSimple2D):
    @pytest.fixture
    def init_env(self):
        env = Sargolini2006(verbose=True, time_step_size=None, agent_step_size=None)
        return [
            env,
        ]

    def test_init_env(self, init_env):
        assert isinstance(init_env[0], Sargolini2006)

    def test_session_plot(self, init_env):
        init_env[0].experiment.plot_trajectory()


class TestHafting2008(TestSargolini2006):
    @pytest.fixture
    def init_env(self):
        env = Hafting2008(
            data_path=None,
            verbose=True,
            # session=session,
            time_step_size=None,
            agent_step_size=None,
        )
        return [
            env,
        ]

    def test_init_env(self, init_env):
        assert isinstance(init_env[0], Hafting2008)


class TestConnectedRooms(TestSimple2D):
    @pytest.fixture
    def init_env(self):
        env_name = "Connected_rooms"
        time_step_size = 0.1  # seg
        agent_step_size = 3

        # Init environment
        env = ConnectedRooms(
            environment_name=env_name,
            time_step_size=time_step_size,
            agent_step_size=agent_step_size,
        )
        return [
            env,
        ]

    def test_init_env(self, init_env):
        assert isinstance(init_env[0], ConnectedRooms)


class TestWernle2018(TestSimple2D):
    @pytest.fixture
    def init_env(self):
        env_name = "MergingRoom"
        time_step_size = 0.2
        agent_step_size = 3
        merging_time = 40
        switch_time = 20
        ((merging_time + switch_time) * 60) / time_step_size

        env = Wernle2018(
            environment_name=env_name,
            merge_time=merging_time,
            switch_time=switch_time,
            time_step_size=time_step_size,
            agent_step_size=agent_step_size,
        )
        return [
            env,
        ]

    def test_init_env(self, init_env):
        assert isinstance(init_env[0], Wernle2018)


class TestMergingRoom(TestSimple2D):
    @pytest.fixture
    def init_env(self):
        env_name = "MergingRoom"
        time_step_size = 0.2
        agent_step_size = 3
        merging_time = 40
        switch_time = 20
        ((merging_time + switch_time) * 60) / time_step_size
        room_width = [-10, 10]
        room_depth = [-10, 10]

        env = MergingRoom(
            arena_x_limits=room_width,
            arena_y_limits=room_depth,
            environment_name=env_name,
            merge_time=merging_time,
            switch_time=switch_time,
            time_step_size=time_step_size,
            agent_step_size=agent_step_size,
        )
        return [
            env,
        ]

    def test_init_env(self, init_env):
        assert isinstance(init_env[0], MergingRoom)


class TestBatchEnvironment(object):
    @pytest.fixture
    def init_env(self):
        env_name = "BatchEnvironment_test"
        batch_size = 16
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
            environment_name=env_name,
            env_class=DiscreteObjectEnvironment,
            batch_size=batch_size,
            arena_x_limits=arena_x_limits,
            arena_y_limits=arena_y_limits,
            arg_env_params=discrete_env_params,
        )
        return [
            env,
        ]

    def test_agent_interaction(self):
        agent_params = parameters.parameters()
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
        env_name = "BatchEnvironment_test"
        batch_size = 16
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
            environment_name=env_name,
            env_class=DiscreteObjectEnvironment,
            batch_size=batch_size,
            arena_x_limits=arena_x_limits,
            arena_y_limits=arena_y_limits,
            arg_env_params=discrete_env_params,
        )
        n_steps = 1
        # Initialize environment
        obs, state = env.reset(random_state=True, custom_state=None)
        for i in range(n_steps):
            while agent.n_walk < agent_params["n_rollout"]:
                # Observe to choose an action
                action = agent.batch_act(obs)
                # Run environment for given action
                obs, state, reward = env.step(action, normalize_step=True)
        env.plot_trajectories()

    def test_init_env(self, init_env):
        assert isinstance(init_env[0], BatchEnvironment)


class TestDiscretizedObjectEnvrionment(TestSimple2D):
    @pytest.fixture
    def init_env(self):
        env_name = "DiscretizedObjectEnvironment_test"
        state_density = 1
        n_objects = 45
        agent_step_size = 1
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
        env = DiscreteObjectEnvironment(
            environment_name=env_name,
            env_class=DiscreteObjectEnvironment,
            arena_x_limits=arena_x_limits,
            arena_y_limits=arena_y_limits,
            state_density=state_density,
            n_objects=n_objects,
            agent_step_size=agent_step_size,
            use_behavioural_data=False,
            data_path=None,
            experiment_class=Sargolini2006Data,
        )
        return [
            env,
        ]

    def test_init_env(self, init_env):
        assert isinstance(init_env[0], DiscreteObjectEnvironment)
