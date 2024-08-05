"""
This file runs a training simulation for the Whittington et al. 2020 agent, the Tolman-Eichenbaum Machine (TEM).
The TEM is a model of the hippocampus that learns to navigate a series of environments and solve a series of tasks.
"""

import os

import numpy as np

from neuralplayground.agents.whittington_2020 import Whittington2020
from neuralplayground.agents.whittington_2020_extras import whittington_2020_parameters as parameters
from neuralplayground.arenas import BatchEnvironment, DiscreteObjectEnvironment
from neuralplayground.backend import SingleSim, tem_training_loop
from neuralplayground.experiments import Sargolini2006Data

# Set the location for saving the results of the simulation
simulation_id = "TEM_var_walks"
save_path = os.path.join(os.getcwd(), simulation_id)
# save_path = os.path.join(os.getcwd(), "examples", "agent_examples", "trained_results")
agent_class = Whittington2020
env_class = BatchEnvironment
training_loop = tem_training_loop

params = parameters.parameters()
full_agent_params = params.copy()

# Set the x and y limits for the arena
arena_x_limits = [
    [-5, 5],
    [-5, 5],
    [-5, 5],
    [-5, 5],
    [-5, 5],
    [-5, 5],
    [-5, 5],
    [-5, 5],
    [-5, 5],
    [-5, 5],
    [-5, 5],
    [-5, 5],
    [-5, 5],
    [-5, 5],
    [-5, 5],
    [-5, 5],
]
arena_y_limits = [
    [-5, 5],
    [-5, 5],
    [-5, 5],
    [-5, 5],
    [-5, 5],
    [-5, 5],
    [-5, 5],
    [-5, 5],
    [-5, 5],
    [-5, 5],
    [-5, 5],
    [-5, 5],
    [-5, 5],
    [-5, 5],
    [-5, 5],
    [-5, 5],
]

# Set parameters for the environment that generates observations
discrete_env_params = {
    "environment_name": "DiscreteObject",
    "state_density": 1 / 2,
    "n_objects": params["n_x"],
    "agent_step_size": 2,  # Note: this must be 1 / state density
    "use_behavioural_data": False,
    "data_path": None,
    "experiment_class": Sargolini2006Data,
}

# Set parameters for the batch environment
env_params = {
    "environment_name": "BatchEnvironment",
    "batch_size": 16,
    "arena_x_limits": arena_x_limits,
    "arena_y_limits": arena_y_limits,
    "env_class": DiscreteObjectEnvironment,
    "arg_env_params": discrete_env_params,
}

# If behavioural data are used, set arena limits to those from Sargolini et al. 2006, reduce state density to 1/4
state_densities = [discrete_env_params["state_density"] for _ in range(env_params["batch_size"])]
if discrete_env_params["use_behavioural_data"]:
    arena_x_limits = [[-50, 50] for _ in range(env_params["batch_size"])]
    arena_y_limits = [[-50, 50] for _ in range(env_params["batch_size"])]
    state_densities = [0.25] * env_params["batch_size"]

room_widths = [int(np.diff(arena_x_limits)[i]) for i in range(env_params["batch_size"])]
room_depths = [int(np.diff(arena_y_limits)[i]) for i in range(env_params["batch_size"])]

# Set parameters for the agent
agent_params = {
    "model_name": "Whittington2020",
    "save_name": str(simulation_id)[4:],
    "params": full_agent_params,
    "batch_size": env_params["batch_size"],
    "room_widths": room_widths,
    "room_depths": room_depths,
    "state_densities": state_densities,
    "use_behavioural_data": discrete_env_params["use_behavioural_data"],
}

# Full model training consists of 20000 episodes
training_loop_params = {"n_episode": 20000, "params": full_agent_params, "random_state": True, "custom_state": [0.0, 0.0]}

# Create the training simulation object
sim = SingleSim(
    simulation_id=simulation_id,
    agent_class=agent_class,
    agent_params=agent_params,
    env_class=env_class,
    env_params=env_params,
    training_loop=training_loop,
    training_loop_params=training_loop_params,
)

# Run the simulation
print("Running sim...")
sim.run_sim(save_path)
print("Sim finished.")
