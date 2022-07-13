import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import random
from sehec.envs.envcore import Environment


class TEMenv(Environment):
    def __init__(self, environment_name="TEMenv", **env_kwargs):
        super().__init__(environment_name, **env_kwargs)
        self.metadata = {"env_kwargs": env_kwargs}
        self.widths = env_kwargs['widths']
        self.agent_step_size = env_kwargs["agent_step_size"]
        self.batch_size = env_kwargs['batch_size']
        self.t_episode = env_kwargs['t_episode']
        self.state_density = env_kwargs['state_density']
        self.world_type = env_kwargs['world_type']
        self.stay_still = env_kwargs['stay_still']
        self.batch_size = 16
        self.s_size = 45
        self.s_size_comp = 10
        self.p_size = env_kwargs['p_size']
        self.g_size = env_kwargs['g_size']
        self.g_init = env_kwargs['g_init']
        self.s_size_comp = env_kwargs['s_size_comp']
        self.n_freq = env_kwargs['n_freq']
        self.n_states = env_kwargs['n_states']

        self.poss_objects = np.zeros(shape=(self.s_size, self.s_size))
        for i in range(self.s_size):
            rand = random.randint(0, self.s_size)
            for j in range(self.s_size):
                if j == rand:
                    self.poss_objects[i][j] = 1

        self.reset()

    def reset(self):
        self.global_steps = 0
        self.history = []
        self.state = [0, 0]
        self.state = np.array(self.state)
        observation = self.state  # make_observation()

        return observation, self.state

    def make_observation(self, pos, objects):
        room_width = np.sqrt(len(objects))
        room_depth = room_width
        resolution_d = int(self.state_density * room_depth)
        resolution_w = int(self.state_density * room_width)
        x_array = np.linspace(-room_width / 2, room_width / 2, num=resolution_d)
        y_array = np.linspace(room_depth / 2, -room_depth / 2, num=resolution_w)
        mesh = np.array(np.meshgrid(x_array, y_array))
        xy_combinations = mesh.T.reshape(-1, 2)

        diff = xy_combinations - pos[np.newaxis, ...]
        dist = np.sum(diff ** 2, axis=1)
        index = np.argmin(dist)
        curr_state = index

        observation = objects[curr_state]

        return observation

    def step(self, actions):
        self.global_steps += 1
        observations = np.zeros(shape=(self.batch_size, self.s_size, self.t_episode))
        new_states = np.zeros(shape=(self.batch_size, 2, self.t_episode))
        rewards = []
        for batch in range(self.batch_size):
            objects = np.zeros(shape=(self.n_states[batch], self.s_size))

            for i in range(self.n_states[batch]):
                rand = random.randint(0, self.s_size - 1)
                objects[i] = self.poss_objects[rand]

            room_width = self.widths[batch]
            room_depth = self.widths[batch]
            for step in range(self.t_episode):
                action = actions[batch, :, step] / np.linalg.norm(actions[batch, :, step])
                new_state = self.state + self.agent_step_size * action
                new_state = np.array([np.clip(new_state[0], a_min=-room_width / 2, a_max=room_width / 2),
                                      np.clip(new_state[1], a_min=-room_depth / 2, a_max=room_depth / 2)])
                reward = 0  # If you get reward, it should be coded here
                transition = {"action": action, "state": self.state, "next_state": new_state,
                              "reward": reward, "step": self.global_steps}
                self.state = new_state
                observation = self.make_observation(new_state, objects)
                if batch == 0:
                    self.history.append(transition)

                observations[batch, :, step] = observation
                new_states[batch, :, step] = new_state
                rewards.append(reward)

        return observations, new_states, rewards

    def plot_trajectory(self, history_data=None, ax=None):
        if history_data is None:
            history_data = self.history
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(8, 6))

        room_width = self.widths[0]
        room_depth = room_width

        ax.plot([-room_width / 2, room_width / 2],
                [-room_depth / 2, -room_depth / 2], "r", lw=2)
        ax.plot([-room_width / 2, room_width / 2],
                [room_depth / 2, room_depth / 2], "r", lw=2)
        ax.plot([-room_width / 2, -room_width / 2],
                [-room_depth / 2, room_depth / 2], "r", lw=2)
        ax.plot([room_width / 2, room_width / 2],
                [-room_depth / 2, room_depth / 2], "r", lw=2)

        state_history = [s["state"] for s in history_data]
        next_state_history = [s["next_state"] for s in history_data]
        starting_point = state_history[0]
        ending_point = next_state_history[-1]
        print(starting_point)

        cmap = mlp.cm.get_cmap("plasma")
        norm = plt.Normalize(0, len(state_history))

        aux_x = []
        aux_y = []
        for i, s in enumerate(state_history):
            x_ = [s[0], next_state_history[i][0]]
            y_ = [s[1], next_state_history[i][1]]
            aux_x.append(s[0])
            aux_y.append(s[1])
            ax.plot(x_, y_, "-", color=cmap(norm(i)), alpha=0.6)

        sc = ax.scatter(aux_x, aux_y, c=np.arange(len(state_history)),
                        vmin=0, vmax=len(state_history), cmap="plasma", alpha=0.6)

        cbar = plt.colorbar(sc, ax=ax)
        cbar.ax.set_ylabel('N steps', rotation=270)

        return ax

    def make_environment(self):
        n_envs = len(self.widths)
        adjs, trans = [], []

        for env in range(n_envs):
            width = self.widths[env]

            if self.world_type == 'square':
                adj, tran = self.square_world(width, self.stay_still)

            else:
                raise ValueError('incorrect world specified')

            adjs.append(adj)
            trans.append(tran)

        return adjs, trans

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
