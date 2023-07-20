import numpy as np

from neuralplayground.agents import Stachenfeld2018, Weber2018
from neuralplayground.arenas import Hafting2008, MergingRoom, Sargolini2006, Simple2D, Wernle2018
from neuralplayground.backend import default_training_loop, episode_based_training_loop

from .cross_comparison import SingleSim

sim1_params = {
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
    "training_loop_params": {"n_steps": 1000},
}

sim2_params = {
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
    "training_loop_params": {"t_episode": 100, "n_episode": 100},
}

sim3_params = {
    "simulation_id": "stachenfeld_2018_in_Sargolini2006",
    "agent_class": Stachenfeld2018,
    "env_class": Sargolini2006,
    "env_params": {
        "use_behavioral_data": True,
        "time_step_size": 0.1,
        "agent_step_size": None,
    },
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
    "training_loop_params": {"t_episode": 100, "n_episode": 100},
}

sim4_params = {
    "simulation_id": "weber_2018_in_Sargolini2006",
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
    "training_loop_params": {"n_steps": 1000},
}


sim5_params = {
    "simulation_id": "stachenfeld_2018_in_Hafting2008",
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
    "training_loop_params": {"t_episode": 100, "n_episode": 100},
}

sim6_params = {
    "simulation_id": "weber_2018_in_Hafting2008",
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
    "training_loop_params": {"n_steps": 1000},
}

sim7_params = {
    "simulation_id": "weber_2018_in_Wernle",
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
    "training_loop_params": {"n_steps": 1000},
}


sim8_params = {
    "simulation_id": "weber_2018_in_Merging_Room",
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
    "training_loop_params": {"n_steps": 1000},
}


# Time in minutes to remove
sim_object1 = SingleSim(**sim1_params)
sim_object2 = SingleSim(**sim2_params)
sim_object3 = SingleSim(**sim3_params)
sim_object4 = SingleSim(**sim4_params)
sim_object5 = SingleSim(**sim5_params)
sim_object6 = SingleSim(**sim6_params)
sim_object7 = SingleSim(**sim7_params)
sim_object8 = SingleSim(**sim8_params)
