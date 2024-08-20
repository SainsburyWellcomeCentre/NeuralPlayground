import os
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from neuralplayground.arenas import Simple2D
from neuralplayground.agents import RandomAgent, LevyFlightAgent, TrajectoryGenerator
from neuralplayground.experiments import Sargolini2006Data
from neuralplayground.agents import TrajectoryGenerator, Burak2009, Sorscher2022exercise, SorscherIdealRNN
from neuralplayground.utils import PlaceCells, get_2d_sort
from neuralplayground.plotting import plot_trajectory_place_cells_activity, plot_ratemaps, compute_ratemaps
from neuralplayground.config import load_plot_config
load_plot_config()
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import os
import pickle
import matplotlib as mpl
import scipy
np.random.seed(0)

# Use GPU if available
device = torch.device("cpu")
agent_step_size = 0.02
agent = TrajectoryGenerator(step_size = agent_step_size)
time_step_size = 0.01


# Init environment
env = Simple2D(time_step_size = time_step_size,
               agent_step_size = agent_step_size,
               arena_x_limits=(-2, 2),
arena_y_limits=(-2, 2))

n_steps = 5000#50000

# Initialize environment
obs, state, crossed = env.reset()
for i in range(n_steps):
    # Observe to choose an action
    action = agent.act(obs,crossed)
    # Run environment for given action
    obs, state, reward, crossed = env.step(action)
ax = env.plot_trajectory()
ax.grid()
plt.show()


print('testing stuff ')


# We'll use a longer sequence just for plotting purposes
# Training will be done with short sequences
sequence_length = 300
batch_size = 4
room_width = 2.2
room_depth = 2.2
# Arena dimensions  Just 2D
room_width = 2.2
room_depth = 2.2

# We'll use a longer sequence just for plotting purposes
# Training will be done with short sequences
sequence_length = 300
batch_size = 4

# Place cells parameters
n_place_cells = 512
place_cell_rf = 0.12
surround_scale = 2.0
periodic = False
difference_of_gaussians = True
place_cells = PlaceCells(Np=n_place_cells,
                         place_cell_rf=place_cell_rf,
                         surround_scale=surround_scale,
                         room_width=room_width,
                         room_depth=room_depth,
                         periodic=periodic,
                         DoG=difference_of_gaussians,
                         device=device)
device = torch.device("cpu")
generator = TrajectoryGenerator(sequence_length, batch_size, room_width, room_depth, device, place_cells=place_cells)
traj = generator.generate_trajectory(room_width, room_depth, batch_size)
x, y = traj["target_x"], traj["target_y"]