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


def main(activation="relu"):

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
                                    learning_rate=learning_rate)

    real_rnn.load_model("tmp_tutorial_model/pre_trained_"+activation)

    res = 50
    n_avg = 100
    Ng = n_grid_cells
    activations, rate_map, g, pos = compute_ratemaps(real_rnn,
                                                     generator,
                                                     sequence_length,
                                                     batch_size,
                                                     room_width,
                                                     room_depth,
                                                     res=res,
                                                     n_avg=n_avg,
                                                     Ng=n_grid_cells)

    # Fourier transform
    Ng = n_grid_cells
    rm_fft_real = np.zeros([Ng, res, res])
    rm_fft_imag = np.zeros([Ng, res, res])

    for i in tqdm(range(Ng)):
        rm_fft_real[i] = np.real(np.fft.fft2(rate_map[i].reshape([res, res])))
        rm_fft_imag[i] = np.imag(np.fft.fft2(rate_map[i].reshape([res, res])))

    rm_fft = rm_fft_real + 1j * rm_fft_imag

    k1 = [3, 0]
    k2 = [2, 3]
    k3 = [-1, 3]
    k4 = k5 = k6 = k1

    freq = 1
    # Equation 39 in Sorscher paper, spatial later assign a sheet position to each neuron
    ks = freq * np.array([k1, k2, k3, k4, k5, k6])
    ks = ks.astype('int')

    modes = np.stack([rm_fft[:, k[0], k[1]] for k in ks])
    phases = [np.angle(mode) for mode in modes]

    N = rate_map.shape[0]
    n = int(np.sqrt(N))
    width = int(np.sqrt(N))
    freq = 1
    X, Y = np.meshgrid(np.arange(width), np.arange(width))
    X = X * 2 * np.pi / width
    Y = Y * 2 * np.pi / width

    s1 = np.zeros(phases[0].shape)
    s2 = np.zeros(phases[0].shape)

    fac = np.sqrt(3) / 2

    for i in range(Ng):
        penalty_1 = np.cos(freq * X - phases[0][i] / fac)
        penalty_2 = np.cos(freq * Y - phases[2][i] / fac)
        penalty_3 = np.cos(freq * (X + Y) - phases[1][i] / fac)
        ind = np.argmax(penalty_1 + penalty_2 + penalty_3 + np.random.randn() / 100)
        s1[i], s2[i] = np.unravel_index([ind], penalty_1.shape)

    total_order = get_2d_sort(s1, s2)
    rm_sort_square = rate_map[total_order.ravel()].reshape([n, n, -1])

    J = real_rnn.recurrent_W.T.detach().cpu().numpy()
    J = J[total_order][:, total_order]
    M = real_rnn.velocity_W.T.detach().cpu().numpy()
    M = M[:, total_order]

    thetas = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    clock_idxs = np.roll([0, 1, 2, 5, 8, 7, 6, 3], 1)
    idx = np.ravel_multi_index((20, 20), (res, res))
    G = (J @ rate_map) > 0

    plt.figure(figsize=(9, 9))
    for i in range(8):
        theta = thetas[i]

        # Get JGMv
        # Get JGMv
        v = np.stack([np.cos(theta), np.sin(theta)])
        Mv = M.T @ v
        GMv = G[:, idx] * Mv
        JGMv = J @ GMv

        # Plot
        plt.subplot(3, 3, clock_idxs[i] + 1)
        print(JGMv.shape)
        im = JGMv.reshape(n, n)
        # im = JGMv[total_order][:, total_order].reshape(n, n)
        im = scipy.ndimage.gaussian_filter(im, (3, 3))
        plt.imshow(im, cmap='RdBu')
        plt.axis('off')
    plt.tight_layout()


if __name__ == "__main__":
    main()