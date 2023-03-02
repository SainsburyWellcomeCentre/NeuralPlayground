# Standard library imports
import numpy as np
import torch
import glob
import matplotlib.pyplot as plt
from importlib import util
from neuralplayground.arenas.torch_TEM_env import TEM_env
from neuralplayground.agents.TEM_extras.TEM_parameters import parameters
import neuralplayground.agents.TEM_extras.TEM_analyse as analyse
import neuralplayground.agents.TEM_extras.TEM_plot as plot

pars_orig = parameters()
params = pars_orig.copy()

room_width = [-5, 5]
room_depth = [-5, 5]
env_name = "env_test"
n_envs = 1
time_step_size = 1
agent_step_size = 1

# Init environment
env = TEM_env(environment_name=env_name,
              arena_x_limits=room_width,
              arena_y_limits=room_depth,
              time_step_size=time_step_size,
              agent_step_size=agent_step_size)

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Choose which trained model to load
date = '2023-03-02' # 2020-10-13 run 0 for successful node agent
run = '7'
index = '1800'
path = '/nfs/nhome/live/lhollingsworth/Documents/NeuralPlayground/NPG/EHC_model_comparison/examples/Summaries2/'
OG_path = '/nfs/nhome/live/lhollingsworth/Documents/NeuralPlayground/Summaries/'

# Load the model: use import library to import module from specified path
model_spec = util.spec_from_file_location("model", path + date + '/torch_run' + run + '/model/TEM_model.py')
model = util.module_from_spec(model_spec)
model_spec.loader.exec_module(model)

# Load the parameters of the model
params = torch.load(path + date + '/torch_run' + run + '/model/params_' + index + '.pt')
# Create a new tem model with the loaded parameters
tem = model.Model(params)
# Load the model weights after training
model_weights = torch.load(path + date + '/torch_run' + run + '/model/tem_' + index + '.pt')
# Set the model weights to the loaded trained model weights
tem.load_state_dict(model_weights)
# Make sure model is in evaluate mode (not crucial because it doesn't currently use dropout or batchnorm layers)
tem.eval()

# Generate test environments
environments = [env.generate_test_environment() for env_i in range(n_envs)]
walk_len = np.median([env[2] * 50 for env in environments]).astype(int)
walks = [env.generate_walks(env_i, walk_len, 1)[0] for env_i in environments]

model_input = [[[[walks[i][j][k]][0] for i in range(len(walks))] for k in range(3)] for j in range(walk_len)]
for i_step, step in enumerate(model_input):
    model_input[i_step][1] = torch.stack(step[1], dim=0)

# Run a forward pass through the model using this data, without accumulating gradients
with torch.no_grad():
    forward = tem(model_input, prev_iter=None)

# Decide whether to include stay-still actions as valid occasions for inference
include_stay_still = True

# Compare trained model performance to a node agent and an edge agent
correct_model, correct_node, correct_edge = analyse.compare_to_agents(forward, tem, environments, include_stay_still=include_stay_still)

# Analyse occurrences of zero-shot inference: predict the right observation arriving from a visited node with a new action
zero_shot = analyse.zero_shot(forward, tem, environments, include_stay_still=include_stay_still)

# Generate occupancy maps: how much time TEM spends at every location
occupation = analyse.location_occupation(forward, tem, environments)

# Generate rate maps
g, p = analyse.rate_map(forward, tem, environments)

# Calculate accuracy leaving from and arriving to each location
from_acc, to_acc = analyse.location_accuracy(forward, tem, environments)

# Choose which environment to plot
env_to_plot = 0
# Set which environments will include shiny objects
shiny_envs = [False, False, False, False]
# And when averaging environments, e.g. for calculating average accuracy, decide which environments to include
envs_to_avg = shiny_envs if shiny_envs[env_to_plot] else [not shiny_env for shiny_env in shiny_envs]

# Plot results of agent comparison and zero-shot inference analysis
filt_size = 41
plt.figure()
plt.plot(analyse.smooth(np.mean(np.array([env for env_i, env in enumerate(correct_model) if envs_to_avg[env_i]]),0)[1:], filt_size), label='tem')
plt.plot(analyse.smooth(np.mean(np.array([env for env_i, env in enumerate(correct_node) if envs_to_avg[env_i]]),0)[1:], filt_size), label='node')
plt.plot(analyse.smooth(np.mean(np.array([env for env_i, env in enumerate(correct_edge) if envs_to_avg[env_i]]),0)[1:], filt_size), label='edge')
plt.ylim(0, 1)
plt.legend()
plt.title('Zero-shot inference: ' + str(np.mean([np.mean(env) for env_i, env in enumerate(zero_shot) if envs_to_avg[env_i]]) * 100) + '%')
plt.savefig('zero_shot.png')

# Plot rate maps for all cells
plot.plot_cells(p[env_to_plot], g[env_to_plot], environments[env_to_plot], n_f_ovc=(params['n_f_ovc'] if 'n_f_ovc' in params else 0), columns = 25)
plt.savefig('rate_maps.png')

# Plot accuracy separated by location
plt.figure()
ax = plt.subplot(1,2,1)
plot.plot_map(environments[env_to_plot], np.array(to_acc[env_to_plot]), ax)
ax.set_title('Accuracy to location')
ax = plt.subplot(1,2,2)
plot.plot_map(environments[env_to_plot], np.array(from_acc[env_to_plot]), ax)
ax.set_title('Accuracy from location')
plt.savefig('accuracy_from.png')

# Plot occupation per location, then add walks on top
ax = plot.plot_map(environments[env_to_plot], np.array(occupation[env_to_plot])/sum(occupation[env_to_plot])*environments[env_to_plot][2],
                   min_val=0, max_val=2, ax=None, shape='square', radius=1/np.sqrt(environments[env_to_plot][2]))
ax = plot.plot_walk(environments[env_to_plot], walks[env_to_plot], ax=ax, n_steps=max(1, int(len(walks[env_to_plot])/500)))
plt.title('Walk and average occupation')
plt.savefig('occupation.png')
plt.show()
