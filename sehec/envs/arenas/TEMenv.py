import matplotlib.pyplot as plt
import numpy as np
from sehec.envs.arenas.simple2d import Simple2D


class TEMenv(Simple2D):
    def __init__(self, environment_name="TEMenv", **env_kwargs):
        super().__init__(environment_name, **env_kwargs)
        self.environment_name = environment_name
        self.width = env_kwargs['room_width']
        self.stay_still = env_kwargs['stay_still']
        self.p_size = env_kwargs['p_size']
        self.g_size = env_kwargs['g_size']
        self.g_init = env_kwargs['g_init']
        self.s_size_comp = env_kwargs['s_size_comp']
        self.n_freq = env_kwargs['n_freq']
        self.n_state = env_kwargs['n_state']
        self.reset()

    def square_world(self, width, stay_still):
        states = int(width ** 2)
        adj = np.zeros((states, states))

        for i in range(states):
            # stay still
            if stay_still:
                adj[i, i] = 1
            # up - down
            if i + width < states:
                adj[i, i + width] = 1
                adj[i + width, i] = 1
                # left - right
            if np.mod(i, width) != 0:
                adj[i, i - 1] = 1
                adj[i - 1, i] = 1

        tran = np.zeros((states, states))
        for i in range(states):
            if sum(adj[i]) > 0:
                tran[i] = adj[i] / sum(adj[i])

        f, ax = plt.subplots(1, 1, figsize=(14, 5))
        ax.imshow(tran, interpolation='nearest')

        f, ax = plt.subplots(1, 1, figsize=(14, 5))
        ax.imshow(adj, interpolation='nearest')

        return adj, tran

    def initialise_hebbian(self):
        a_rnn = np.zeros((self.p_size, self.p_size))
        a_rnn_inv = np.zeros((self.p_size, self.p_size))

        return a_rnn, a_rnn_inv

    def initialise_variables(self):
        gs = np.maximum(np.random.randn(self.g_size) * self.g_init, 0)
        x_s = np.zeros(self.s_size_comp * self.n_freq)

        visited = np.zeros(self.n_state)

        return gs, x_s, visited
