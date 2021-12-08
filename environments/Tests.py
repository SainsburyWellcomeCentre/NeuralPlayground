import sys
sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
from environments.env_list.simple2d import Simple2D
from model.basic import ExcitInhibitoryplastic

# Create an env
room_width = 0.5
room_depth = 0.5
env_name = "env_example"
time_step_size = 1
agent_step_size = 0.02

# Init environment
env = Simple2D(environment_name=env_name,
               room_width = room_width,
               room_depth = room_depth,
               time_step_size = time_step_size,
               agent_step_size = agent_step_size)

# Setting of the figure
exc_eta_1 = 6.7e-5
inh_eta_1 = 2.7e-4
model_name_1 = "model_example"
sigma_exc_0 = 0.05
sigma_inh_0 = 0.1
sigma_exc_2 = 0.05
sigma_inh_2 =0.1
Ne_1 = 4900
Ni_1 = 1225
alpha_inh_1 = 1e-260
alpha_exc_1 = 1e-260

agent = ExcitInhibitoryplastic(step=agent_step_size,model_name=model_name_1,
                               exc_eta=exc_eta_1, inh_eta=inh_eta_1,
                               sigma_exc=sigma_exc_0, sigma_inh=sigma_inh_0,
                               sigma_exc_1=sigma_exc_2, sigma_inh_1=sigma_inh_2,
                               Ne=Ne_1, Ni=Ni_1,
                               alpha_inh=alpha_inh_1, alpha_exc=alpha_exc_1)

n_steps = 100
agent.reset()
# Initialize environment
obs, state = env.reset()
for i in range(n_steps):
    # Observe to choose an action
    action = agent.act(obs)
    rate =  agent.update(action,env.room_width, env.room_depth)
    # Run environment for given action
    obs, state, reward = env.step(action)

ax = env.plot_trajectory()
fontsize = 16
ax.grid()
ax.legend(fontsize=fontsize, loc="upper left")
ax.set_xlabel("width", fontsize=fontsize)
ax.set_ylabel("depth", fontsize=fontsize)
plt.show()
X=agent.plot_rate(room_width, room_depth)
