# Standard Imports
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np
import pickle
import torch
import importlib.util

# NeuralPlayground Imports
from neuralplayground.arenas.discritized_objects import DiscreteObjectEnvironment
from neuralplayground.arenas.batch_environment import BatchEnvironment
from neuralplayground.agents.whittington_2020 import Whittington2020
import neuralplayground.agents.whittington_2020_extras.whittington_2020_analyse as analyse

# NeuralPlayground Experiment Imports
from neuralplayground.arenas.hafting_2008 import Hafting2008
from neuralplayground.experiments import Sargolini2006Data

# Select trained model
date = '2023-05-17'
run = '0'
index = '19999'
base_path = '/nfs/nhome/live/lhollingsworth/Documents/NeuralPlayground/NPG/EHC_model_comparison'
npg_path = '/nfs/nhome/live/lhollingsworth/Documents/NeuralPlayground/NPG/EHC_model_comparison/examples'
base_win_path = 'H:/Documents/PhD/NeuralPlayground'
win_path = 'H:/Documents/PhD/NeuralPlayground/NPG/NeuralPlayground/examples'
# Load the model: use import library to import module from specified path
model_spec = importlib.util.spec_from_file_location("model", win_path + '/Summaries/' + date + '/torch_run' + run + '/script/whittington_2020_model.py')
model = importlib.util.module_from_spec(model_spec)
model_spec.loader.exec_module(model)

# Load the parameters of the model
params = torch.load(win_path + '/Summaries/' + date + '/torch_run' + run + '/model/params_' + index + '.pt')
# Create a new tem model with the loaded parameters
tem = model.Model(params)
# Load the model weights after training
model_weights = torch.load(win_path + '/Summaries/' + date + '/torch_run' + run + '/model/tem_' + index + '.pt')
# Set the model weights to the loaded trained model weights
tem.load_state_dict(model_weights)
# Make sure model is in evaluate mode (not crucial because it doesn't currently use dropout or batchnorm layers)
tem.eval()

# Initialise environment parameters
batch_size = 16
arena_x_limits = [[-5,5], [-4,4], [-5,5], [-6,6], [-4,4], [-5,5], [-6,6], [-5,5], [-4,4], [-5,5], [-6,6], [-5,5], [-4,4], [-5,5], [-6,6], [-5,5]]
arena_y_limits = [[-5,5], [-4,4], [-5,5], [-6,6], [-4,4], [-5,5], [-6,6], [-5,5], [-4,4], [-5,5], [-6,6], [-5,5], [-4,4], [-5,5], [-6,6], [-5,5]]
# arena_x_limits = [[-20,20], [-20,20], [-15,15], [-10,10], [-20,20], [-20,20], [-15,15], [-10,10], [-20,20], [-20,20], [-15,15], [-10,10], [-20,20], [-20,20], [-15,15], [-10,10]]
# arena_y_limits = [[-4,4],   [-2,2],   [-2,2],   [-1,1],   [-4,4],   [-2,2],   [-2,2],   [-1,1],   [-4,4],   [-2,2],   [-2,2],   [-1,1],   [-4,4],   [-2,2],   [-2,2],   [-1,1]]
env_name = "env_example"
mod_name = "SimpleTEM"
time_step_size = 1
state_density = 1
agent_step_size = 1/state_density
n_objects = 45

# Init simple 2D environment with discrtised objects
env_class = DiscreteObjectEnvironment
env = BatchEnvironment(environment_name=env_name,
                       env_class=DiscreteObjectEnvironment,
                       batch_size=batch_size,
                       arena_x_limits=arena_x_limits,
                       arena_y_limits=arena_y_limits,
                       state_density=state_density,
                       n_objects=n_objects,
                       agent_step_size=agent_step_size,
                       use_behavioural_data=False,
                       data_path=None,
                       experiment_class=Sargolini2006Data)

# Init TEM agent
agent = Whittington2020(model_name=mod_name,
                        params=params,
                        batch_size=batch_size,
                        room_widths=env.room_widths,
                        room_depths=env.room_depths,
                        state_densities=env.state_densities,
                        use_behavioural_data=False)

# # Run around environment
# observation, state = env.reset(random_state=True, custom_state=None)
# while agent.n_walk < 5000:
#     if agent.n_walk % 100 == 0:
#         print(agent.n_walk)
#     action = agent.batch_act(observation)
#     observation, state = env.step(action, normalize_step=True)
# model_input, history, environments = agent.collect_final_trajectory()
# environments = [env.collect_environment_info(model_input, history, environments)]

# # Save environments and model_input using pickle
# with open('NPG_environments.pkl', 'wb') as f:
#     pickle.dump(environments, f)
# with open('NPG_model_input.pkl', 'wb') as f:
#     pickle.dump(model_input, f)

# Load environments and model_input using pickle
with open('NPG_environments.pkl', 'rb') as f:
    environments = pickle.load(f)
with open('NPG_model_input.pkl', 'rb') as f:
    model_input = pickle.load(f)

with torch.no_grad():
    forward = tem(model_input, prev_iter=None)
include_stay_still = False
shiny_envs = [False, False, False, False]
env_to_plot = 0
envs_to_avg = shiny_envs if shiny_envs[env_to_plot] else [not shiny_env for shiny_env in shiny_envs]

correct_model, correct_node, correct_edge = analyse.compare_to_agents(forward, tem, environments, include_stay_still=include_stay_still)
zero_shot = analyse.zero_shot(forward, tem, environments, include_stay_still=include_stay_still)
occupation = analyse.location_occupation(forward, tem, environments)
g, p = analyse.rate_map(forward, tem, environments)
from_acc, to_acc = analyse.location_accuracy(forward, tem, environments)

# Plot rate maps for grid or place cells
agent.plot_rate_map(g)

# Plot results of agent comparison and zero-shot inference analysis
filt_size = 41
plt.figure()
plt.plot(analyse.smooth(np.mean(np.array([env for env_i, env in enumerate(correct_model) if envs_to_avg[env_i]]),0)[1:], filt_size), label='tem')
plt.plot(analyse.smooth(np.mean(np.array([env for env_i, env in enumerate(correct_node) if envs_to_avg[env_i]]),0)[1:], filt_size), label='node')
plt.plot(analyse.smooth(np.mean(np.array([env for env_i, env in enumerate(correct_edge) if envs_to_avg[env_i]]),0)[1:], filt_size), label='edge')
plt.ylim(0, 1)
plt.legend()
plt.title('Zero-shot inference: ' + str(np.mean([np.mean(env) for env_i, env in enumerate(zero_shot) if envs_to_avg[env_i]]) * 100) + '%')

# plt.show()