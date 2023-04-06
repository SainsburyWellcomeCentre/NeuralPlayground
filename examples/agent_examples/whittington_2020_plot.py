# Standard Imports
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np
import torch
import importlib.util

# NeuralPlayground Imports
from neuralplayground.arenas.discritized_objects import DiscreteObjectEnvironment
from neuralplayground.arenas.batch_environment import BatchEnvironment
from neuralplayground.arenas.hafting_2008 import Hafting2008
from neuralplayground.agents.whittington_2020 import Whittington2020
import neuralplayground.agents.whittington_2020_extras.whittington_2020_parameters as parameters
import neuralplayground.agents.whittington_2020_extras.whittington_2020_analyse as analyse
import neuralplayground.agents.whittington_2020_extras.whittington_2020_plot as plot

# Select trained model
date = '2023-03-28'
run = '1'
index = '9999'
torch_path = '/nfs/nhome/live/lhollingsworth/Documents/NeuralPlayground'
# Load the model: use import library to import module from specified path
model_spec = importlib.util.spec_from_file_location("model", torch_path + '/Summaries/' + date + '/torch_run' + run + '/script/whittington_2020_model.py')
model = importlib.util.module_from_spec(model_spec)
model_spec.loader.exec_module(model)

# Load the parameters of the model
params = torch.load(torch_path + '/Summaries/' + date + '/torch_run' + run + '/model/params_' + index + '.pt')
# Create a new tem model with the loaded parameters
tem = model.Model(params)
# Load the model weights after training
model_weights = torch.load(torch_path + '/Summaries/' + date + '/torch_run' + run + '/model/tem_' + index + '.pt')
# Set the model weights to the loaded trained model weights
tem.load_state_dict(model_weights)
# Make sure model is in evaluate mode (not crucial because it doesn't currently use dropout or batchnorm layers)
tem.eval()

# Initialise environment parameters
batch_size = 16
arena_x_limits = [[10,10], [10,10], [10,10], [10,10], [10,10], [10,10], [10,10], [10,10], [10,10], [10,10], [10,10], [10,10], [10,10], [10,10], [10,10], [10,10]]
arena_y_limits = [[10,10], [10,10], [10,10], [10,10], [10,10], [10,10], [10,10], [10,10], [10,10], [10,10], [10,10], [10,10], [10,10], [10,10], [10,10], [10,10]]
env_name = "env_example"
mod_name = "SimpleTEM"
time_step_size = 1
state_density = 1
agent_step_size = 1/state_density
n_objects = 45

# Init simple 2D environment with discrtised objects
env_class = DiscreteObjectEnvironment
env = BatchEnvironment(environment_name=env_name,
                       env_class=env_class,
                       batch_size=batch_size,
                       arena_x_limits=arena_x_limits,
                       arena_y_limits=arena_y_limits,
                       state_density=state_density,
                       n_objects=n_objects,
                       agent_step_size=agent_step_size)

# Init TEM agent
agent = Whittington2020(model_name=mod_name,
                        params=params,
                        batch_size=batch_size,
                        room_widths=env.room_widths,
                        room_depths=env.room_depths,
                        state_densities=env.state_densities)

# Run around environment
observation, state = env.reset(random_state=True, custom_state=None)
while agent.n_walk < 5000:
    action = agent.batch_act(observation)
    observation, state = env.step(action, normalize_step=True)
model_input = agent.final_model_input
environments = agent.collect_environment_info(model_input)

with torch.no_grad():
    forward = tem(model_input, prev_iter=None)


