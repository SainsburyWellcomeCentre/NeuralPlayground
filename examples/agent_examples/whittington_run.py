import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import os, shutil
import tqdm
from neuralplayground.arenas.torch_TEM_env import TEM_env
from neuralplayground.agents.whittington_2020 import TEM
import neuralplayground.agents.TEM_extras.TEM_parameters as parameters

pars_orig = parameters.parameters()
params = pars_orig.copy()

room_width = [-5, 5]
room_depth = [-5, 5]
env_name = "env_example"
mod_name = "TorchTEMTest"
time_step_size = 1
agent_step_size = 1

# Init environment
env = TEM_env(environment_name=env_name,
              arena_x_limits=room_width,
              arena_y_limits=room_depth,
              time_step_size=time_step_size,
              agent_step_size=agent_step_size)
agent = TEM(model_name=mod_name, params=params)

walk, obs, actions = env.batch_reset(normalize_step=False, random_state=True)
for i in range(params['train_it']):
    print(i)
    agent.act(i, walk, obs, actions)
    walk, obs, actions = env.batch_step(normalize_step=False)
    

ax = env.plot_batch_trajectory()
fontsize = 18
ax.grid()
ax.set_xlabel("width", fontsize=fontsize)
ax.set_ylabel("depth", fontsize=fontsize)
plt.savefig('trajectory.png')
