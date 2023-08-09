import numpy as np

from neuralplayground.agents import AgentCore, Stachenfeld2018, Weber2018
from neuralplayground.arenas import Environment, Hafting2008, MergingRoom, Sargolini2006, Simple2D, Wernle2018
from neuralplayground.backend import default_training_loop, episode_based_training_loop

from .simulation_manager import SingleSim

weber_in_2d_params = {
    "simulation_id": "weber_2018_in_simple2D",
    "agent_class": Weber2018,
    "agent_params": {
        "exc_eta": 2e-4,
        "inh_eta": 8e-4,
        "model_name": "weber_2018",
        "sigma_exc": np.array([0.05, 0.05]),
        "sigma_inh": np.array([0.1, 0.1]),
        "Ne": 4900,
        "Ni": 1225,
        "Nef": 1,
        "Nif": 1,
        "alpha_i": 1,
        "alpha_e": 1,
        "we_init": 1.0,
        "wi_init": 1.5,
        "agent_step_size": 0.1,
        "resolution": 100,
        "ro": 1,
        "room_width": 20.0,
        "room_depth": 20.0,
        "disable_tqdm": True,
    },
    "env_class": Simple2D,
    "env_params": {
        "arena_x_limits": np.array([-10, 10]),
        "arena_y_limits": np.array([-10, 10]),
        "env_name": "env_example",
        "time_step_size": 1,
        "agent_step_size": 0.5,
    },
    "training_loop": default_training_loop,
    "training_loop_params": {"n_steps": 10000},
}

stachenfeld_in_2d_params = {
    "simulation_id": "stachenfeld_2018_in_simple2D",
    "agent_class": Stachenfeld2018,
    "env_class": Simple2D,
    "env_params": {
        "arena_x_limits": [-6, 6],
        "arena_y_limits": [-6, 6],
        "env_name": "env_example",
        "time_step_size": 0.2,
        "agent_step_size": 1,
    },
    "agent_params": {
        "discount": 0.9,
        "threshold": 1e-6,
        "lr_td": 1e-2,
        "state_density": 1,
        "room_width": 12,
        "room_depth": 12,
        "twoD": True,
    },
    "training_loop": episode_based_training_loop,
    "training_loop_params": {"t_episode": 1000, "n_episode": 100},
}

stachenfeld_in_sargolini_params = {
    "simulation_id": "stachenfeld_2018_in_sargolini2006",
    "env_class": Sargolini2006,
    "env_params": {
        "use_behavioral_data": True,
        "time_step_size": 0.1,
        "agent_step_size": None,
    },
    "agent_class": Stachenfeld2018,
    "agent_params": {
        "discount": 0.9,
        "threshold": 1e-6,
        "lr_td": 1e-2,
        "state_density": 0.1,
        "room_width": 100,
        "room_depth": 100,
        "twoD": True,
    },
    "training_loop": episode_based_training_loop,
    "training_loop_params": {"t_episode": 1000, "n_episode": 100},
}

weber_in_sargolini_params = {
    "simulation_id": "weber_2018_in_sargolini2006",
    "env_class": Sargolini2006,
    "env_params": {
        "use_behavioral_data": True,
        "time_step_size": 0.1,
        "agent_step_size": None,
    },
    "agent_class": Weber2018,
    "agent_params": {
        "exc_eta": 2e-4,
        "inh_eta": 8e-4,
        "model_name": "weber_2018",
        "sigma_exc": np.array([0.05, 0.05]),
        "sigma_inh": np.array([0.1, 0.1]),
        "Ne": 4900,
        "Ni": 1225,
        "Nef": 1,
        "Nif": 1,
        "alpha_i": 1,
        "alpha_e": 1,
        "we_init": 1.0,
        "wi_init": 1.5,
        "agent_step_size": 0.1,
        "resolution": 100,
        "ro": 1,
        "room_width": 100,
        "room_depth": 100,
        "disable_tqdm": True,
    },
    "training_loop": default_training_loop,
    "training_loop_params": {"n_steps": 10000},
}


stachenfeld_in_hafting_params = {
    "simulation_id": "stachenfeld_2018_in_hafting2008",
    "agent_class": Stachenfeld2018,
    "env_class": Hafting2008,
    "env_params": {
        "use_behavioral_data": True,
        "time_step_size": 0.1,
        "agent_step_size": None,
    },
    "agent_params": {
        "discount": 0.9,
        "threshold": 1e-6,
        "lr_td": 1e-2,
        "state_density": 1,
        "room_width": 20.0,
        "room_depth": 2,
        "twoD": True,
    },
    "training_loop": episode_based_training_loop,
    "training_loop_params": {"t_episode": 1000, "n_episode": 100},
}

weber_in_hafting_params = {
    "simulation_id": "weber_2018_in_hafting2008",
    "agent_class": Weber2018,
    "agent_params": {
        "exc_eta": 2e-4,
        "inh_eta": 8e-4,
        "model_name": "weber_2018",
        "sigma_exc": np.array([0.05, 0.05]),
        "sigma_inh": np.array([0.1, 0.1]),
        "Ne": 4900,
        "Ni": 1225,
        "Nef": 1,
        "Nif": 1,
        "alpha_i": 1,
        "alpha_e": 1,
        "we_init": 1.0,
        "wi_init": 1.5,
        "agent_step_size": 0.1,
        "resolution": 100,
        "ro": 1,
        "room_width": 20.0,
        "room_depth": 2,
        "disable_tqdm": True,
    },
    "env_class": Hafting2008,
    "env_params": {
        "use_behavioral_data": True,
        "time_step_size": 0.1,
        "agent_step_size": None,
    },
    "training_loop": default_training_loop,
    "training_loop_params": {"n_steps": 10000},
}

stachenfeld_in_wernle_params_1 = {
    "simulation_id": "stachenfeld_2018_in_wernle_2",
    "agent_class": Stachenfeld2018,
    "agent_params": {
        "discount": 0.9,
        "threshold": 1e-6,
        "lr_td": 1e-2,
        "state_density": 0.1,
        "room_width": 100,
        "room_depth": 100,
        "twoD": True,
    },
    "env_class": Wernle2018,
    "env_params": {"merge_time": 20, "switch_time": 10, "time_step_size": 0.2, "agent_step_size": 3},
    "training_loop": episode_based_training_loop,
    "training_loop_params": {"t_episode": 1000, "n_episode": 100},
}

stachenfeld_in_wernle_params_2 = {
    "simulation_id": "stachenfeld_2018_in_wernle_2",
    "agent_class": Stachenfeld2018,
    "agent_params": {
        "discount": 0.9,
        "threshold": 1e-6,
        "lr_td": 1e-2,
        "state_density": 0.1,
        "room_width": 100,
        "room_depth": 100,
        "twoD": True,
    },
    "env_class": Wernle2018,
    "env_params": {"merge_time": 20, "switch_time": 10, "time_step_size": 0.2, "agent_step_size": 3},
    "training_loop": default_training_loop,
    "training_loop_params": {"n_steps": 10000},
}


weber_in_wernle_params = {
    "simulation_id": "weber_2018_in_wernle",
    "agent_class": Weber2018,
    "agent_params": {
        "exc_eta": 2e-4,
        "inh_eta": 8e-4,
        "model_name": "weber_2018",
        "sigma_exc": np.array([0.05, 0.05]),
        "sigma_inh": np.array([0.1, 0.1]),
        "Ne": 4900,
        "Ni": 1225,
        "Nef": 1,
        "Nif": 1,
        "alpha_i": 1,
        "alpha_e": 1,
        "we_init": 1.0,
        "wi_init": 1.5,
        "agent_step_size": 0.1,
        "resolution": 100,
        "ro": 1,
        "room_width": 200.0,
        "room_depth": 200.0,
        "disable_tqdm": True,
    },
    "env_class": Wernle2018,
    "env_params": {"merge_time": 20, "switch_time": 10, "time_step_size": 0.2, "agent_step_size": 3},
    "training_loop": default_training_loop,
    "training_loop_params": {"n_steps": 10000},
}


weber_in_merging_room_params = {
    "simulation_id": "weber_2018_in_merging_room",
    "agent_class": Weber2018,
    "agent_params": {
        "exc_eta": 2e-4,
        "inh_eta": 8e-4,
        "model_name": "weber_2018",
        "sigma_exc": np.array([0.05, 0.05]),
        "sigma_inh": np.array([0.1, 0.1]),
        "Ne": 4900,
        "Ni": 1225,
        "Nef": 1,
        "Nif": 1,
        "alpha_i": 1,
        "alpha_e": 1,
        "we_init": 1.0,
        "wi_init": 1.5,
        "agent_step_size": 0.1,
        "resolution": 100,
        "ro": 1,
        "room_width": 20.0,
        "room_depth": 20.0,
        "disable_tqdm": True,
    },
    "env_class": MergingRoom,
    "env_params": {
        "arena_x_limits": [-10, 10],
        "arena_y_limits": [-10, 10],
        "time_step_size": 0.2,
        "agent_step_size": 10,
        "merge_time": 270.0,
        "switch_time": 270.0,
    },
    "training_loop": default_training_loop,
    "training_loop_params": {"n_steps": 10000},
}


stachenfeld_in_merging_room_1 = {
    "simulation_id": "stachenfeld_2018_in_merging_room_1",
    "agent_class": Stachenfeld2018,
    "agent_params": {
        "discount": 0.9,
        "threshold": 1e-6,
        "lr_td": 1e-2,
        "state_density": 0.1,
        "room_width": 100,
        "room_depth": 100,
        "twoD": True,
    },
    "env_class": MergingRoom,
    "env_params": {
        "arena_x_limits": [-10, 10],
        "arena_y_limits": [-10, 10],
        "time_step_size": 0.2,
        "agent_step_size": 10,
        "merge_time": 270.0,
        "switch_time": 270.0,
    },
    "training_loop": default_training_loop,
    "training_loop_params": {"n_steps": 10000},
}

stachenfeld_in_merging_room_2 = {
    "simulation_id": "stachenfeld_2018_in_merging_room_2",
    "agent_class": Stachenfeld2018,
    "agent_params": {
        "discount": 0.9,
        "threshold": 1e-6,
        "lr_td": 1e-2,
        "state_density": 0.1,
        "room_width": 100,
        "room_depth": 100,
        "twoD": True,
    },
    "env_class": MergingRoom,
    "env_params": {
        "arena_x_limits": [-10, 10],
        "arena_y_limits": [-10, 10],
        "time_step_size": 0.2,
        "agent_step_size": 10,
        "merge_time": 270.0,
        "switch_time": 270.0,
    },
    "training_loop": episode_based_training_loop,
    "training_loop_params": {"t_episode": 1000, "n_episode": 100},
}


coreEnv_test_1 = {
    "simulation_id": "coreenv_test_1",
    "agent_class": Weber2018,
    "agent_params": {
        "exc_eta": 2e-4,
        "inh_eta": 8e-4,
        "model_name": "weber_2018",
        "sigma_exc": np.array([0.05, 0.05]),
        "sigma_inh": np.array([0.1, 0.1]),
        "Ne": 4900,
        "Ni": 1225,
        "Nef": 1,
        "Nif": 1,
        "alpha_i": 1,
        "alpha_e": 1,
        "we_init": 1.0,
        "wi_init": 1.5,
        "agent_step_size": 0.1,
        "resolution": 100,
        "ro": 1,
        "room_width": 20.0,
        "room_depth": 20.0,
        "disable_tqdm": True,
    },
    "env_class": Environment,
    "env_params": {
        "time_step_size": 0.2,
        "environment_name": "test",
    },
    "training_loop": default_training_loop,
    "training_loop_params": {"n_steps": 10000},
}


coreEnv_test_2 = {
    "simulation_id": "coreenv_test_2",
    "agent_class": Weber2018,
    "agent_params": {
        "exc_eta": 2e-4,
        "inh_eta": 8e-4,
        "model_name": "weber_2018",
        "sigma_exc": np.array([0.05, 0.05]),
        "sigma_inh": np.array([0.1, 0.1]),
        "Ne": 4900,
        "Ni": 1225,
        "Nef": 1,
        "Nif": 1,
        "alpha_i": 1,
        "alpha_e": 1,
        "we_init": 1.0,
        "wi_init": 1.5,
        "agent_step_size": 0.1,
        "resolution": 100,
        "ro": 1,
        "room_width": 20.0,
        "room_depth": 20.0,
        "disable_tqdm": True,
    },
    "env_class": Environment,
    "env_params": {
        "time_step_size": 0.2,
        "environment_name": "test",
    },
    "training_loop": episode_based_training_loop,
    "training_loop_params": {"t_episode": 1000, "n_episode": 100},
}

coreEnv_test_3 = {
    "simulation_id": "coreenv_test_3",
    "agent_class": Stachenfeld2018,
    "agent_params": {
        "discount": 0.9,
        "threshold": 1e-6,
        "lr_td": 1e-2,
        "state_density": 0.1,
        "room_width": 100,
        "room_depth": 100,
        "twoD": True,
    },
    "env_class": Environment,
    "env_params": {
        "time_step_size": 0.2,
        "environment_name": "test",
    },
    "training_loop": episode_based_training_loop,
    "training_loop_params": {"t_episode": 1000, "n_episode": 100},
}

coreEnv_test_4 = {
    "simulation_id": "coreenv_test_4",
    "agent_class": Stachenfeld2018,
    "agent_params": {
        "discount": 0.9,
        "threshold": 1e-6,
        "lr_td": 1e-2,
        "state_density": 0.1,
        "room_width": 100,
        "room_depth": 100,
        "twoD": True,
    },
    "env_class": Environment,
    "env_params": {
        "time_step_size": 0.2,
        "environment_name": "test",
    },
    "training_loop": default_training_loop,
    "training_loop_params": {"n_steps": 10000},
}


coreAgentEnv_test = {
    "simulation_id": "coreagentenv_test",
    "agent_class": AgentCore,
    "agent_params": {
        "model_name": "agent_test",
    },
    "env_class": Environment,
    "env_params": {
        "time_step_size": 0.2,
        "environment_name": "test",
    },
    "training_loop": default_training_loop,
    "training_loop_params": {"n_steps": 10000},
}

coreAgentEnv_test_2 = {
    "simulation_id": "coreagentenv_test_2",
    "agent_class": AgentCore,
    "agent_params": {
        "model_name": "agent_test",
    },
    "env_class": Environment,
    "env_params": {
        "time_step_size": 0.2,
        "environment_name": "test",
    },
    "training_loop": episode_based_training_loop,
    "training_loop_params": {"t_episode": 1000, "n_episode": 100},
}


coreAgent_test_1 = {
    "simulation_id": "coreagent_test_1",
    "agent_class": AgentCore,
    "agent_params": {
        "model_name": "agent_test",
    },
    "env_class": Simple2D,
    "env_params": {
        "arena_x_limits": [-6, 6],
        "arena_y_limits": [-6, 6],
        "env_name": "env_example",
        "time_step_size": 0.2,
        "agent_step_size": 1,
    },
    "training_loop": default_training_loop,
    "training_loop_params": {"n_steps": 10000},
}

coreAgent_test_2 = {
    "simulation_id": "coreagent_test_2",
    "agent_class": AgentCore,
    "agent_params": {
        "model_name": "agent_test",
    },
    "env_class": Simple2D,
    "env_params": {
        "arena_x_limits": [-6, 6],
        "arena_y_limits": [-6, 6],
        "env_name": "env_example",
        "time_step_size": 0.2,
        "agent_step_size": 1,
    },
    "training_loop": episode_based_training_loop,
    "training_loop_params": {"t_episode": 1000, "n_episode": 100},
}


coreAgent_test_3 = {
    "simulation_id": "coreagent_test_3",
    "agent_class": AgentCore,
    "agent_params": {
        "model_name": "agent_test",
    },
    "env_class": Hafting2008,
    "env_params": {
        "use_behavioral_data": True,
        "time_step_size": 0.1,
        "agent_step_size": None,
    },
    "training_loop": default_training_loop,
    "training_loop_params": {"n_steps": 10000},
}

coreAgent_test_4 = {
    "simulation_id": "coreagent_test_4",
    "agent_class": AgentCore,
    "agent_params": {
        "model_name": "agent_test",
    },
    "env_class": Wernle2018,
    "env_params": {"merge_time": 20, "switch_time": 10, "time_step_size": 0.2, "agent_step_size": 3},
    "training_loop": default_training_loop,
    "training_loop_params": {"n_steps": 10000},
}


coreAgent_test_5 = {
    "simulation_id": "coreagent_test_5",
    "agent_class": AgentCore,
    "agent_params": {
        "model_name": "agent_test",
    },
    "env_class": MergingRoom,
    "env_params": {
        "arena_x_limits": [-10, 10],
        "arena_y_limits": [-10, 10],
        "time_step_size": 0.2,
        "agent_step_size": 10,
        "merge_time": 270.0,
        "switch_time": 270.0,
    },
    "training_loop": default_training_loop,
    "training_loop_params": {"n_steps": 10000},
}


# Time in minutes to remove
weber_in_wernle = SingleSim(**weber_in_wernle_params)
weber_in_merging_room = SingleSim(**weber_in_merging_room_params)
weber_in_sargolini = SingleSim(**weber_in_sargolini_params)
weber_in_hafting = SingleSim(**weber_in_hafting_params)
weber_in_2d = SingleSim(**weber_in_2d_params)

stachenfeld_in_wernle_1 = SingleSim(**stachenfeld_in_wernle_params_1)
stachenfeld_in_wernle_2 = SingleSim(**stachenfeld_in_wernle_params_2)
stachenfeld_in_merging_room_1 = SingleSim(**stachenfeld_in_merging_room_1)
stachenfeld_in_merging_room_2 = SingleSim(**stachenfeld_in_merging_room_2)
stachenfeld_in_sargolini = SingleSim(**stachenfeld_in_sargolini_params)
stachenfeld_in_hafting = SingleSim(**stachenfeld_in_hafting_params)
stachenfeld_in_2d = SingleSim(**stachenfeld_in_2d_params)

coreEnv_test_1 = SingleSim(**coreEnv_test_1)
coreEnv_test_2 = SingleSim(**coreEnv_test_2)
coreEnv_test_3 = SingleSim(**coreEnv_test_3)
coreEnv_test_4 = SingleSim(**coreEnv_test_4)

coreAgentEnv_test = SingleSim(**coreAgentEnv_test)
coreAgentEnv_test_2 = SingleSim(**coreAgentEnv_test_2)

coreAgent_test_1 = SingleSim(**coreAgent_test_1)
coreAgent_test_2 = SingleSim(**coreAgent_test_2)
coreAgent_test_3 = SingleSim(**coreAgent_test_3)
coreAgent_test_4 = SingleSim(**coreAgent_test_4)
coreAgent_test_5 = SingleSim(**coreAgent_test_5)
