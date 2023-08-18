"""
This file runs a training simulation for the Whittington et al. 2020 agent, the Tolman-Eichenbaum Machine (TEM).
The TEM is a model of the hippocampus that learns to navigate a series of environments and solve a series of tasks.
"""

import os

import numpy as np

from neuralplayground.agents import Whittington2020
from neuralplayground.agents.whittington_2020_extras import whittington_2020_parameters as parameters
from neuralplayground.arenas import BatchEnvironment, DiscreteObjectEnvironment
from neuralplayground.backend import SingleSim, tem_training_loop
from neuralplayground.experiments import Sargolini2006Data

simulation_id = "TEM_custom_sim"
save_path = os.path.join(os.getcwd(), "examples", "agent_examples", "results_sim")
# save_path = os.path.join(os.getcwd(), "examples", "agent_examples", "trained_results")
agent_class = Whittington2020
env_class = BatchEnvironment
training_loop = tem_training_loop

params = parameters.parameters()
full_agent_params = params.copy()

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
    "n_objects": params["n_x"],
    "agent_step_size": 1,
    "use_behavioural_data": True,
    "data_path": None,
    "experiment_class": Sargolini2006Data,
}

env_params = {
    "environment_name": "BatchEnvironment",
    "batch_size": 16,
    "arena_x_limits": arena_x_limits,
    "arena_y_limits": arena_y_limits,
    "env_class": DiscreteObjectEnvironment,
    "arg_env_params": discrete_env_params,
}

if discrete_env_params["use_behavioural_data"]:
    arena_x_limits = [[-50, 50] for _ in range(env_params["batch_size"])]
    arena_y_limits = [[-50, 50] for _ in range(env_params["batch_size"])]
    state_densities = [0.25] * env_params["batch_size"]

room_widths = [int(np.diff(arena_x_limits)[i]) for i in range(env_params["batch_size"])]
room_depths = [int(np.diff(arena_y_limits)[i]) for i in range(env_params["batch_size"])]

agent_params = {
    "model_name": "Whittington2020",
    "params": full_agent_params,
    "batch_size": env_params["batch_size"],
    "room_widths": room_widths,
    "room_depths": room_depths,
    "state_densities": state_densities,
    "use_behavioural_data": discrete_env_params["use_behavioural_data"],
}

# Full model training consists of 20000 episodes
training_loop_params = {"n_episode": 10, "params": full_agent_params}

sim = SingleSim(
    simulation_id=simulation_id,
    agent_class=agent_class,
    agent_params=agent_params,
    env_class=env_class,
    env_params=env_params,
    training_loop=training_loop,
    training_loop_params=training_loop_params,
)

# print(sim)
print("Running sim...")
sim.run_sim(save_path)
print("Sim finished.")
