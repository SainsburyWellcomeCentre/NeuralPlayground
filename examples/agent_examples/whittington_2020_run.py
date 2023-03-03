import matplotlib.pyplot as plt
from neuralplayground.arenas.batch_environment import BatchEnvironment
from neuralplayground.agents.whittington_2020 import Whittington2020
import neuralplayground.agents.whittington_2020_extras.whittington_2020_parameters as parameters

pars_orig = parameters.parameters()
params = pars_orig.copy()

room_width = [-5, 5]
room_depth = [-5, 5]
env_name = "env_example"
mod_name = "TorchTEMTest"
time_step_size = 1
agent_step_size = 1
state_density = params['state_density']

# Init environment
env = TEM_env(environment_name=env_name,
              arena_x_limits=room_width,
              arena_y_limits=room_depth,
              state_density=state_density,
              time_step_size=time_step_size,
              agent_step_size=agent_step_size)
agent = Whittington2020(model_name=mod_name, params=params,
            room_width=room_width, room_depth=room_depth,
            state_density=state_density)

positions, states = env.batch_reset(normalize_step=False, random_state=True)
for i in range(params['train_it']):
    while agent.n_walk < params['n_rollout']:
        actions = agent.act(positions)
        positions, states = env.batch_step(actions)
    agent.update()

ax = env.plot_batch_trajectory()
fontsize = 18
ax.grid()
ax.set_xlabel("width", fontsize=fontsize)
ax.set_ylabel("depth", fontsize=fontsize)
plt.savefig('trajectory.png')
