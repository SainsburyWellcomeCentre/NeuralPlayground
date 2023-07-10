import numpy as np

from neuralplayground.agents import Stachenfeld2018, Weber2018
from neuralplayground.arenas import Simple2D
from neuralplayground.backend import default_training_loop, episode_based_training_loop

from .cross_comparison import SingleSim

# Experiment 1: Weber 2018
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
    "training_loop_params": {"n_steps": 100},
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
    "training_loop_params": {"t_episode": 10, "n_episode": 10},
}


sim_object1 = SingleSim(**sim1_params)
sim_object2 = SingleSim(**sim2_params)
