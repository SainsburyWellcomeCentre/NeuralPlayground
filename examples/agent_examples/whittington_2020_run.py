import matplotlib.pyplot as plt
from neuralplayground.arenas.simple2d import Simple2D
from neuralplayground.arenas.discritized_objects import DiscreteObjectEnvironment
from neuralplayground.arenas.batch_environment import BatchEnvironment
from neuralplayground.arenas.hafting_2008 import Hafting2008
from neuralplayground.arenas.sargolini_2006 import Sargolini2006
from neuralplayground.agents.whittington_2020 import Whittington2020
import neuralplayground.agents.whittington_2020_extras.whittington_2020_parameters as parameters

pars_orig = parameters.parameters()
params = pars_orig.copy()

batch_size = 16
arena_x_limits = [[-5,5], [-4,4], [-5,5], [-6,6], [-4,4], [-5,5], [-6,6], [-5,5], [-4,4], [-5,5], [-6,6], [-5,5], [-4,4], [-5,5], [-6,6], [-5,5]]
arena_y_limits = [[-5,5], [-4,4], [-5,5], [-6,6], [-4,4], [-5,5], [-6,6], [-5,5], [-4,4], [-5,5], [-6,6], [-5,5], [-4,4], [-5,5], [-6,6], [-5,5]]
env_name = "env_example"
mod_name = "SimpleTEM"
time_step_size = 1
state_density = 1
agent_step_size = 1/state_density
n_objects = 45

# env = Hafting2008(agent_step_size=agent_step_size,
#                   time_step_size=time_step_size,
#                   use_behavioral_data=False)

# arena_x_limits = env.arena_x_limits
# arena_y_limits = env.arena_y_limits
env_class = DiscreteObjectEnvironment

# Init environment
env = BatchEnvironment(environment_name=env_name,
                       env_class=env_class,
                       batch_size=batch_size,
                       arena_x_limits=arena_x_limits,
                       arena_y_limits=arena_y_limits,
                       state_density=state_density,
                       n_objects=n_objects,
                       agent_step_size=agent_step_size)
agent = Whittington2020(model_name=mod_name,
                        params=params,
                        batch_size=batch_size,
                        room_widths=env.room_widths,
                        room_depths=env.room_depths,
                        state_densities=env.state_densities)

observation, state = env.reset(random_state=True, custom_state=None)
for i in range(params['train_it']):
    while agent.n_walk < params['n_rollout']:
        actions = agent.batch_act(observation)
        observation, state = env.step(actions, normalize_step=True)
    agent.update()

# ax = env.plot_batch_trajectory()
# fontsize = 18
# ax.grid()
# ax.set_xlabel("width", fontsize=fontsize)
# ax.set_ylabel("depth", fontsize=fontsize)
# plt.savefig('trajectory.png')
