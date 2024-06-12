""" Agent implementation of the Burak2009 model.
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000291
The code was adapted from the publicly available one originally implemented in Matlab.
"""
import numpy as np
from scipy.fft import fft2, fftshift, ifft2
from tqdm import tqdm

from neuralplayground.agents import AgentCore


class Burak2009(AgentCore):
    def __init__(
        self,
        model_name: str = "Idealized_rnn_grid_cells",
        agent_step_size: float = 0.1,
        n_neurons: int = 2**7,  # Number of neurons
        tau: float = 5,  # Neuron time-constant (ms)
        lambda_: float = 13,  # Equation (3)
        beta: float = 3 / 13**2,  # Equation (3)
        alphabar: float = 1.05,  # alphabar = gamma/beta from Equation (3)
        abar: float = 1,  # a should be <= alphabar^2. Equation (3)
        wtphase: int = 2,  # wtphase is 'l' from Equation (2)
        alpha: float = 1,  # The velocity gain from Equation (4)
        time_step_dt=0.5,  # Default timestep (ms)
        room_width=2.2,
        room_depth=2.2,
        offset_weights=False,
        **mod_kwargs,
    ):
        mod_kwargs["agent_step_size"] = agent_step_size
        super().__init__(model_name, **mod_kwargs)
        self.agent_step_size = agent_step_size
        self.n_neurons = n_neurons  # this should be the square of a number
        self.tau = tau
        self.lambda_ = lambda_
        self.beta = beta
        self.alphabar = alphabar
        self.abar = abar
        self.wtphase = wtphase
        self.alpha = alpha
        self.time_step_dt = time_step_dt
        self.offset_weights = offset_weights
        self._initialize_fourier_weights()
        self._initialize_explicit_weights()

    def _initialize_explicit_weights(self):
        """ This is from Sorscher equations 35, 36 abd 38 """
        self.k0 = np.array([1, 0])
        self.k60 = np.array([0.5, np.sqrt(3) / 2])
        self.k120 = np.array([-0.5, np.sqrt(3) / 2])
        self.k_vec = np.stack([self.k0, self.k60, self.k120], axis=0)
        self.L = np.sqrt(self.n_neurons).astype(int)

        Jij = np.zeros((self.n_neurons, self.n_neurons))
        grid_location = np.arange(1, self.L+1)
        print("Building recurrent matrix")
        sheet_locations = []
        for i in range(self.L):
            for j in range(self.L):
                sheet_locations.append(np.array([grid_location[i], grid_location[j]]))
        sheet_locations = np.stack(sheet_locations, axis=0)
        Mx = np.mod(sheet_locations[:, 1], 2)*((-1)**(sheet_locations[:, 0]))
        My = np.mod(sheet_locations[:, 0], 2)*((-1)**(sheet_locations[:, 1]))
        Mixy = np.stack([Mx, My], axis=1)
        for i in range(self.n_neurons):
            si = sheet_locations[i, :]
            for j in range(self.n_neurons):
                sj = sheet_locations[j, :]
                if self.offset_weights:
                    s_diff = si - sj - Mixy[j, :]
                else:
                    s_diff = si - sj
                Jij[i, j] = self.weight_function(s_diff[:, np.newaxis])
        self.Jij = Jij
        self.Mixy = Mixy
        self.bi = np.ones((self.n_neurons, 1))*0.1
        self.sheet_locations = sheet_locations
        print("debug")

    def rate_update(self, rates, velocity):
        matrix_product = self.Jij @ rates
        velocity_product = self.Mixy @ velocity
        new_rates = matrix_product + velocity_product #+ self.bi
        return npRelu(new_rates)
        #return np.clip(new_rates, 0, 10)
        #return npsigmoid(new_rates)

    def weight_function(self, x):
        # Make sure x.shape = (dim, 1), eq 37 in Sorscher
        inner = 2*np.pi/self.L*(self.k_vec @ x)
        element_wise_cos = np.cos(inner)
        return np.sum(element_wise_cos)


    def _initialize_fourier_weights(self):
        # Padding for convolutions
        self.big = 2 * self.n_neurons
        dim = self.n_neurons // 2

        # Initial activity
        self.initial_activity = np.zeros((self.n_neurons, self.n_neurons))

        # Envelope and weight matrix parameters
        x = np.arange(-self.n_neurons // 2, self.n_neurons // 2)
        lx = len(x)
        xbar = np.sqrt(self.beta) * x

        # Center surround, locally inhibitory, weight matrix - Equation (3)
        self.filt = self.abar * np.exp(
            -self.alphabar * (np.outer(np.ones(lx), xbar**2) + np.outer(xbar**2, np.ones(lx)))
        ) - np.exp(-1 * (np.outer(np.ones(lx), xbar**2) + np.outer(xbar**2, np.ones(lx))))

        # Envelope function that determines the global feedforward input - Equation (5)
        self.venvelope = np.exp(
            -4
            * (np.outer(x**2, np.ones(self.n_neurons)) + np.outer(np.ones(self.n_neurons), x**2))
            / (self.n_neurons / 2) ** 2
        )
        # shifted weight matrices
        self.frshift = np.roll(self.filt, self.wtphase, axis=1)
        self.flshift = np.roll(self.filt, -self.wtphase, axis=1)
        self.fdshift = np.roll(self.filt, self.wtphase, axis=0)
        self.fushift = np.roll(self.filt, -self.wtphase, axis=0)

        self.ftu = fft2(self.fushift, (self.big, self.big))
        self.ftd = fft2(self.fdshift, (self.big, self.big))
        self.ftl = fft2(self.flshift, (self.big, self.big))
        self.ftr = fft2(self.frshift, (self.big, self.big))

        self.ftu_small = fft2(fftshift(self.fushift))
        self.ftd_small = fft2(fftshift(self.fdshift))
        self.ftl_small = fft2(fftshift(self.flshift))
        self.ftr_small = fft2(fftshift(self.frshift))

        # Block matrices used for identifying all neurons of one preferred firing direction
        self.typeL = np.tile([[1, 0], [0, 0]], (dim, dim))
        self.typeR = np.tile([[0, 0], [0, 1]], (dim, dim))
        self.typeU = np.tile([[0, 1], [0, 0]], (dim, dim))
        self.typeD = np.tile([[0, 0], [1, 0]], (dim, dim))

    def ideal_grid_cells(self, periodic_boundary=False):
        if not periodic_boundary:
            n_iters = 500
        else:
            n_iters = 1000

        # Initial movement condition
        self.theta_v = np.pi / 5
        self.left = -np.sin(self.theta_v)
        self.right = np.sin(self.theta_v)
        self.up = -np.cos(self.theta_v)
        self.down = np.cos(self.theta_v)
        self.vel = 0

        current_rate = self.initial_activity.copy()
        for iter in range(n_iters):
            if periodic_boundary and iter == 800:
                # For some reason you don't get hexagons if you have no envelope for the entire training
                self.venvelope = np.ones((self.n_neurons, self.n_neurons))
            current_rate = self.ideal_update(current_rate)
        self.grid_cell_rate = current_rate
        return self.grid_cell_rate

    def ideal_update(self, current_rate):
        # Break global input into its directional components
        rfield = self.venvelope * (
            (1 + self.vel * self.right) * self.typeR
            + (1 + self.vel * self.left) * self.typeL
            + (1 + self.vel * self.up) * self.typeU
            + (1 + self.vel * self.down) * self.typeD
        )

        # Convolute pupolation activity with shifted symmetric weight matrices
        convolution = np.real(
            ifft2(
                fft2(current_rate * self.typeR, (self.big, self.big)) * self.ftr
                + fft2(current_rate * self.typeL, (self.big, self.big)) * self.ftl
                + fft2(current_rate * self.typeD, (self.big, self.big)) * self.ftd
                + fft2(current_rate * self.typeU, (self.big, self.big)) * self.ftu
            )
        )

        # Add feedforward input to the shifted population activity
        rfield += convolution[
            self.n_neurons // 2 : self.big - self.n_neurons // 2, self.n_neurons // 2 : self.big - self.n_neurons // 2
        ]

        # Neural transfer function (Rectified Linear Unit)
        fr = np.maximum(rfield, 0)

        r_new = np.minimum(10, (self.time_step_dt / self.tau) * (5 * fr - current_rate) + current_rate)
        return r_new

    def simulate_base_trajectory(self, sequence_length=100000):
        # Let's simulate trajectories in the same way so we get the same results as in the paper
        enclosure_radius = 2 * 100  # Two meters
        temp_velocity = np.random.rand() / 2
        position_x = np.zeros(sequence_length)
        position_y = np.zeros(sequence_length)
        headDirection = np.zeros(sequence_length)
        position_x[0] = 0
        position_y[0] = 0
        headDirection[0] = np.random.rand() * 2 * np.pi

        for i in tqdm(range(sequence_length)):
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

        return position_x, position_y, headDirection

    def path_neural_activity(self, position_x, position_y, headDirection, grid_cell_rate, periodic_boundary=False):
        increment = 1
        r = grid_cell_rate
        sampling_length = len(position_x)
        single_neuron_response = np.zeros(sampling_length)
        single_neuron = [self.n_neurons // 2, self.n_neurons // 2]

        for iter in tqdm(range(sampling_length - 20)):
            theta_v = headDirection[increment]
            vel = np.sqrt(
                (position_x[increment] - position_x[increment - 1]) ** 2
                + (position_y[increment] - position_y[increment - 1]) ** 2
            )

            left = -np.cos(theta_v)
            right = np.cos(theta_v)
            up = np.sin(theta_v)
            down = -np.sin(theta_v)

            increment += 1

            # Break feedforward input into its directional components
            # Equation (4)
            rfield = self.venvelope * (
                (1 + self.alpha * vel * right) * self.typeR
                + (1 + self.alpha * vel * left) * self.typeL
                + (1 + self.alpha * vel * up) * self.typeU
                + (1 + self.alpha * vel * down) * self.typeD
            )

            # Convolute population activity with shifted symmetric weights.
            # real() is implemented for octave compatibility
            convolution = np.real(
                ifft2(
                    fft2(r * self.typeR) * self.ftr_small
                    + fft2(r * self.typeL) * self.ftl_small
                    + fft2(r * self.typeD) * self.ftd_small
                    + fft2(r * self.typeU) * self.ftu_small
                )
            )

            # Add feedforward inputs to the shifted population activity to
            # yield the new population activity.
            rfield += convolution

            # Neural Transfer Function
            fr = np.where(rfield > 0, rfield, 0)

            # Neuron dynamics (Eq. 1)
            r_old = r
            r_new = np.minimum(10, (self.time_step_dt / self.tau) * (5 * fr - r_old) + r_old)
            r = r_new

            # Track single neuron response
            if fr[single_neuron[0], single_neuron[1]] > 0:
                single_neuron_response[increment] = 1

        return single_neuron_response, r


def npRelu(x):
    return np.maximum(0, x)


def npsigmoid(z):
    return 1/(1 + np.exp(-z))


if __name__ == "__main__":
    n_neurons = 24 ** 2
    ideal_rnn = Burak2009(n_neurons=n_neurons)
    periodic_grid_cell = ideal_rnn.ideal_grid_cells(periodic_boundary=True)
    iters = 1000
    rates = np.zeros((n_neurons, 1))
    # Random initial rates
    # rates = np.random.normal(size=(n_neurons, 1))
    velocity = np.zeros((2, 1))
    for i in tqdm(range(iters)):
        rates = ideal_rnn.rate_update(rates, velocity=velocity)
