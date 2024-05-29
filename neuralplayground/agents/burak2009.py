""" Agent implementation of the Burak2009 model.
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000291
The code was adapted from the publicly available one originally implemented in Matlab.
"""
import numpy as np
from scipy.fft import fft2, ifft2

from neuralplayground.agents import AgentCore


class Burak2009(AgentCore):
    def __init__(
        self,
        model_name: str = "Ideallize",
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
        **mod_kwargs,
    ):
        mod_kwargs["agent_step_size"] = agent_step_size
        super().__init__(model_name, **mod_kwargs)
        self.agent_step_size = agent_step_size
        self.n_neurons = n_neurons
        self.tau = tau
        self.lambda_ = lambda_
        self.beta = beta
        self.alphabar = alphabar
        self.abar = abar
        self.wtphase = wtphase
        self.alpha = alpha
        self.time_step_dt = time_step_dt

    def non_periodic_grid_cells(self, sequence_length):
        # Padding for convolutions
        self.big = 2 * self.n_neurons
        dim = self.n_neurons / 2

        # Initial activity
        self.initial_activity = np.zeros((sequence_length, self.n_neurons))

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

        # Block matrices used for identifying all neurons of one preferred firing direction
        self.typeL = np.tile([[1, 0], [0, 0]], (dim, dim))
        self.typeR = np.tile([[0, 0], [0, 1]], (dim, dim))
        self.typeU = np.tile([[0, 1], [0, 0]], (dim, dim))
        self.typeD = np.tile([[0, 0], [1, 0]], (dim, dim))

        # Initial movement condition
        self.theta_v = np.pi / 5
        self.left = -np.sin(self.theta_v)
        self.right = np.sin(self.theta_v)
        self.up = -np.cos(self.theta_v)
        self.down = np.cos(self.theta_v)
        self.vel = 0

        current_rate = self.initial_activity.copy()
        for iter in range(500):
            current_rate = self.update(current_rate)
        self.grid_cell_rate = current_rate
        return self.grid_cell_rate

    def update(self, current_rate):
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

        # Neural transfer function
        fr = np.maximum(rfield, 0)

        r_new = np.minimum(10, (self.time_step_dt / self.tau) * (5 * fr - current_rate) + current_rate)
        return r_new


if __name__ == "__main__":
    pass
