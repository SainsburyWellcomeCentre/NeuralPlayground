import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.fft import fft2, ifft2
from scipy.sparse import lil_matrix

from neuralplayground.vendored.trajectory_generator import TrajectoryGenerator


def simulate(filename, simulate_non_periodic, use_spiking):
    # Define default timestep (ms)
    dt = 0.5

    # ---- Warning ----

    # Parameters (likely from an associated paper)
    n = 2**7  # Number of neurons
    tau = 5  # Neuron time-constant (ms)
    lambda_ = 13  # Equation (3)
    beta = 3 / lambda_**2  # Equation (3)
    alphabar = 1.05  # alphabar = gamma/beta from Equation (3)
    abar = 1  # a should be <= alphabar^2. Equation (3)
    wtphase = 2  # wtphase is 'l' from Equation (2)
    alpha = 1  # The velocity gain from Equation (4)

    # Implement the logic for loading data from filename based on its format (if needed)
    spikes = gc_non_periodic(filename, n, tau, dt, beta, alphabar, abar, wtphase, alpha, use_spiking)
    return spikes


def gc_non_periodic(filename, n, tau, dt, beta, alphabar, abar, wtphase, alpha, useSpiking):
    sequence_length = 100000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = TrajectoryGenerator(sequence_length, batch_size=1, room_width=4, room_depth=4, device=device, place_cells=None)
    traj = generator.generate_trajectory(room_width=4, room_depth=4, batch_size=1)
    # If no data is loaded, use random trajectories.
    enclosure_radius = 2 * 100  # Two meters
    temp_velocity = np.random.rand() / 2
    position_x = np.zeros(sequence_length)
    position_y = np.zeros(sequence_length)
    headDirection = np.zeros(sequence_length)
    position_x[0] = 0
    position_y[0] = 0
    headDirection[0] = np.random.rand() * 2 * np.pi
    traj_x = traj["target_x"][0, :]
    traj_y = traj["target_y"][0, :]
    traj["target_hd"][0, :]

    for i in range(sequence_length):
        # max acceleration is .1 cm/ms^2
        temp_rand = np.clip(np.random.normal(0, 0.05), -0.2, 0.2)

        # max velocity is .5 cm/ms
        temp_velocity = np.clip(temp_velocity + temp_rand, 0, 0.25)

        left_or_right = np.random.choice([-1, 1])

        while (
            np.sqrt(
                (position_x[i - 1] + np.cos(headDirection[i - 1]) * temp_velocity) ** 2
                + (position_y[i - 1] + np.sin(headDirection[i - 1]) * temp_velocity) ** 2
            )
            > enclosure_radius
        ):
            headDirection[i - 1] += left_or_right * np.pi / 100

        position_x[i] = position_x[i - 1] + np.cos(headDirection[i - 1]) * temp_velocity
        position_y[i] = position_y[i - 1] + np.sin(headDirection[i - 1]) * temp_velocity
        headDirection[i] = (headDirection[i - 1] + (np.random.rand() - 0.5) / 5 * np.pi / 2) % (2 * np.pi)

    sampling_length = len(position_x)
    # HeadDirection var doesn't care about the range of angles apparently

    # plt.plot(position_x, position_y, label="matlab code")
    plt.plot(traj_x, traj_y, label="python code")
    plt.legend()
    plt.show()

    print("synthetic data generated")

    # if file_load:
    #     # linearly interpolate data to scale to .5 ms
    #     if dt != .5:
    #         if dt < .1:
    #             # If data is too fine, then downsample to make computations faster
    #             position_x = position_x[::int(np.floor(.5 / dt))]
    #             position_y = position_y[::int(np.floor(.5 / dt))]
    #             dt = int(np.floor(.5 / dt)) * dt
    #         dt = round(dt * 10) / 10
    #
    #         interp_func_x = interp1d(np.arange(len(position_x)), position_x, kind='linear')
    #         interp_func_y = interp1d(np.arange(len(position_y)), position_y, kind='linear')
    #         new_indices = np.linspace(0, len(position_x) - 1, int(len(position_x) * (dt / 0.5)))
    #         position_x = interp_func_x(new_indices)
    #         position_y = interp_func_y(new_indices)
    #
    #         dt = .5
    #
    #     sampling_length = len(position_x)
    #     # Add in head directions
    #     headDirection = np.zeros(sampling_length)
    #     for i in range(sampling_length - 1):
    # headDirection[i] = np.mod(np.arctan2(position_y[i+1] - position_y[i], position_x[i+1] - position_x[i]), 2 * np.pi)
    #     headDirection[sampling_length - 1] = headDirection[sampling_length - 2]

    # ----------------------
    # INITIALIZE VARIABLES
    # ----------------------

    big = 2 * n
    dim = n // 2

    # Initial population activity
    r = np.zeros((n, n))
    rfield = r.copy()
    r.copy()

    spikes = [lil_matrix((n, n)) for _ in range(sampling_length)]

    np.zeros(sampling_length)
    [n // 2, n // 2]

    # Envelope and weight matrix parameters
    x = np.arange(-n // 2, n // 2)
    lx = len(x)
    xbar = np.sqrt(beta) * x

    # Center surround, locally inhibitory, weight matrix - Equation (3)
    filt = abar * np.exp(-alphabar * (np.outer(np.ones(lx), xbar**2) + np.outer(xbar**2, np.ones(lx)))) - np.exp(
        -1 * (np.outer(np.ones(lx), xbar**2) + np.outer(xbar**2, np.ones(lx)))
    )

    # Envelope function that determines the global feedforward input - Equation (5)
    venvelope = np.exp(-4 * (np.outer(x**2, np.ones(n)) + np.outer(np.ones(n), x**2)) / (n / 2) ** 2)

    # shifted weight matrices
    frshift = np.roll(filt, wtphase, axis=1)
    flshift = np.roll(filt, -wtphase, axis=1)
    fdshift = np.roll(filt, wtphase, axis=0)
    fushift = np.roll(filt, -wtphase, axis=0)

    ftu = fft2(fushift, (big, big))
    ftd = fft2(fdshift, (big, big))
    ftl = fft2(flshift, (big, big))
    ftr = fft2(frshift, (big, big))

    # Block matrices used for identifying all neurons of one preferred firing direction
    typeL = np.tile([[1, 0], [0, 0]], (dim, dim))
    typeR = np.tile([[0, 0], [0, 1]], (dim, dim))
    typeU = np.tile([[0, 1], [0, 0]], (dim, dim))
    typeD = np.tile([[0, 0], [1, 0]], (dim, dim))

    # Initial movement condition
    theta_v = np.pi / 5
    left = -np.sin(theta_v)
    right = np.sin(theta_v)
    up = -np.cos(theta_v)
    down = np.cos(theta_v)
    vel = 0

    for iter in range(500):
        # Break global input into its directional components
        rfield = venvelope * (
            (1 + vel * right) * typeR + (1 + vel * left) * typeL + (1 + vel * up) * typeU + (1 + vel * down) * typeD
        )

        # Convolute pupolation activity with shifted symmetric weight matrices
        convolution = np.real(
            ifft2(
                fft2(r * typeR, (big, big)) * ftr
                + fft2(r * typeL, (big, big)) * ftl
                + fft2(r * typeD, (big, big)) * ftd
                + fft2(r * typeU, (big, big)) * ftu
            )
        )

        # Add feedforward input to the shifted population activity
        rfield += convolution[n // 2 : big - n // 2, n // 2 : big - n // 2]

        # Neural transfer function
        fr = np.maximum(rfield, 0)

        # Neuron dynamics (equation 1)
        r_old = r
        r_new = np.minimum(10, (dt / tau) * (5 * fr - r_old) + r_old)
        r = r_new

    return spikes


# Placeholder functions for gc_periodic and gc_non_periodic
# You'll need to implement these functions based on the original MATLAB code


if __name__ == "__main__":
    # Example usage
    spikes = simulate("test_file", False, False)  # Replace with your actual filename
    # agent = Burak2009()
    # grid_cell = agent.ideal_grid_cells(periodic_boundary=False)
    # print("debugging")
