import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import os
import pickle
import scipy

from neuralplayground.experiments import Sargolini2006Data
from neuralplayground.agents import TrajectoryGenerator, Burak2009, Sorscher2022, Sorscher2022exercise
from neuralplayground.utils import PlaceCells, get_2d_sort
from neuralplayground.plotting import plot_trajectory_place_cells_activity, plot_ratemaps, compute_ratemaps
from neuralplayground.config import load_plot_config


def main(activation="tanh"):

    print("Pre-training the network")
    print("Activation function: ", activation)


    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Arena dimensions  Just 2D
    room_width = 2.2
    room_depth = 2.2

    # We'll use a longer sequence just for plotting purposes
    # Training will be done with short sequences
    sequence_length = 20
    batch_size = 256

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

    # Real RNN parameters
    n_grid_cells = 24 ** 2  # needs to be the square of a number for later analysis
    learning_rate = 5e-4
    training_steps = int(1e5)
    weight_decay = 1e-4

    generator = TrajectoryGenerator(sequence_length, batch_size, room_width, room_depth, device,
                                    place_cells=place_cells)
    traj = generator.generate_trajectory(room_width, room_depth, batch_size)

    gen = generator.get_batch_generator()

    # real_rnn = Sorscher2022(Ng=n_grid_cells,
    #                         Np=n_place_cells,
    #                         sequence_length=sequence_length,
    #                         weight_decay=weight_decay,
    #                         place_cells=place_cells,
    #                         activation=activation,
    #                         learning_rate=learning_rate)

    real_rnn = Sorscher2022exercise(Ng=n_grid_cells,
                                    Np=n_place_cells,
                                    sequence_length=sequence_length,
                                    weight_decay=weight_decay,
                                    place_cells=place_cells,
                                    activation=activation,
                                    learning_rate=learning_rate,
                                    device=device)

    loss_hist, pos_err_hist = real_rnn.train_RNN(gen, training_steps)

    # Save the model
    save_path = "tmp_tutorial_model/long_pre_trained_"+activation+".tar"
    real_rnn.save_model(save_path)


if __name__ == "__main__":
    main()