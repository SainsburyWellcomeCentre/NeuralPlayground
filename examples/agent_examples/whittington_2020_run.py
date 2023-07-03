"""
Run file for the Tolman-Eichenbaum Machine (TEM) model from Whittington et al. 2020. An example setup is provided, with
TEM learning to predict upcoming sensory stimulus in a range of 16 square environments of varying sizes.
"""

# Standard Imports
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# NeuralPlayground Arena Imports
from neuralplayground.arenas.discritized_objects import DiscreteObjectEnvironment
from neuralplayground.arenas.batch_environment import BatchEnvironment
from neuralplayground.arenas.hafting_2008 import Hafting2008
from neuralplayground.arenas.sargolini_2006 import BasicSargolini2006
from neuralplayground.agents.whittington_2020 import Whittington2020

# NeuralPlayground Agent Imports
import neuralplayground.agents.whittington_2020_extras.whittington_2020_parameters as parameters

# NeuralPlayground Experiment Imports
from neuralplayground.experiments import Sargolini2006Data

# Initialise TEM Parameters
pars_orig = parameters.parameters()
params = pars_orig.copy()

# Initialise environment parameters
batch_size = 16
# arena_x_limits = [[-5,5], [-4,4], [-5,5], [-6,6], [-4,4], [-5,5], [-6,6], [-5,5], [-4,4], [-5,5], [-6,6], [-5,5], [-4,4], [-5,5], [-6,6], [-5,5]]
# arena_y_limits = [[-5,5], [-4,4], [-5,5], [-6,6], [-4,4], [-5,5], [-6,6], [-5,5], [-4,4], [-5,5], [-6,6], [-5,5], [-4,4], [-5,5], [-6,6], [-5,5]]
arena_x_limits = [[-20,20], [-20,20], [-15,15], [-10,10], [-20,20], [-20,20], [-15,15], [-10,10], [-20,20], [-20,20], [-15,15], [-10,10], [-20,20], [-20,20], [-15,15], [-10,10]]
arena_y_limits = [[-4,4],   [-2,2],   [-2,2],   [-1,1],   [-4,4],   [-2,2],   [-2,2],   [-1,1],   [-4,4],   [-2,2],   [-2,2],   [-1,1],   [-4,4],   [-2,2],   [-2,2],   [-1,1]]
env_name = "Hafting2008"
mod_name = "SimpleTEM"
time_step_size = 1
state_density = 1/10
agent_step_size = 1/state_density
n_objects = 45

# # Init environment from Hafting 2008 (optional, if chosen, comment out the )
# env = Hafting2008(agent_step_size=agent_step_size,
#                   time_step_size=time_step_size,
#                   use_behavioral_data=False)

# # Init simple 2D (batched) environment with discrtised objects
# env_class = DiscreteObjectEnvironment

# Init environment from Sargolini, with behavioural data instead of random walk
env = BatchEnvironment(environment_name=env_name,
                       env_class=DiscreteObjectEnvironment,
                       batch_size=batch_size,
                       arena_x_limits=arena_x_limits,
                       arena_y_limits=arena_y_limits,
                       state_density=state_density,
                       n_objects=n_objects,
                       agent_step_size=agent_step_size,
                       use_behavioural_data=True,
                       data_path=None,
                       experiment_class=Sargolini2006Data)

# Init TEM agent
agent = Whittington2020(model_name=mod_name,
                        params=params,
                        batch_size=batch_size,
                        room_widths=env.room_widths,
                        room_depths=env.room_depths,
                        state_densities=env.state_densities)

# Reset environment and begin training (random_state=True is currently necessary)
observation, state = env.reset(random_state=True, custom_state=None)
for i in range(3):
    print("Iteration: ", i)
    while agent.n_walk < params['n_rollout']:
        print("Walk: ", agent.n_walk)
        actions = agent.batch_act(observation)
        observation, state = env.step(actions, normalize_step=True)
    agent.update()

# Plot most recent trajectory of the first environment in batch
ax = env.plot_trajectory()
fontsize = 18
ax.grid()
ax.set_xlabel("width", fontsize=fontsize)
ax.set_ylabel("depth", fontsize=fontsize)
plt.savefig('trajectory.png')
plt.show()
