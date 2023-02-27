import numpy as np
import torch
import matplotlib.pyplot as plt
import os, shutil
import tqdm
from neuralplayground.arenas.torch_TEM_env import TEM_env
from neuralplayground.agents.torch_TEM import Model
from neuralplayground.agents.TEM_extras.torch_TEM_parameters import parameters
from neuralplayground.agents.TEM_extras.torch_TEM_utils import *

pars_orig = parameters()
params = pars_orig.copy()

room_width = [-10, 10]
room_depth = [-10, 10]
env_name = "env_example"
time_step_size = 1
agent_step_size = 1

# Init environment
env = TEM_env(environment_name=env_name,
              arena_x_limits=room_width,
              arena_y_limits=room_depth,
              time_step_size=time_step_size,
              agent_step_size=agent_step_size)
agent = Model(params)

for i in range(params['train_it']):
    walk, obs, actions = env.batch_reset(normalize_step=False, random_state=True)
    for j in range(params['n_walks'][i]):
        agent.act(i, j, walk, obs, actions)
        walk, obs, actions = env.batch_step(normalize_step=False)
        print('yess')

ax = env.plot_batch_trajectory()
fontsize = 18
ax.grid()
ax.set_xlabel("width", fontsize=fontsize)
ax.set_ylabel("depth", fontsize=fontsize)
plt.show()
