import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import random

from sehec.envs.envcore import Environment
import sehec.models.TEM as model


class TEMenv(Environment):
    def __init__(self, environment_name="TEMenv", **env_kwargs):
        super().__init__(environment_name, **env_kwargs)
        self.metadata = {"env_kwargs": env_kwargs}
        self.pars = env_kwargs
        self.reset()

    def reset(self):
        self.global_steps = 0
        self.history = []
        self.state = [0, 0]
        self.state = np.array(self.state)
        observation = self.state

        return observation, self.state

    def make_observation(self):
        return self.state

    def step(self, policy):
        # Step through environment depending on given action policy
        self.global_steps += 1
        observations = np.zeros(shape=(self.pars['batch_size'], 2, self.pars['t_episode']))
        new_states = np.zeros(shape=(self.pars['batch_size'], 2, self.pars['t_episode']))
        rewards = []

        actions = np.zeros((self.pars['batch_size'], 2, self.pars['t_episode']))
        direcs = np.zeros(shape=(self.pars['batch_size'], 4, self.pars['t_episode']))

        observation = [0, 0]

        for batch in range(self.pars['batch_size']):
            room_width = self.pars['widths'][batch]
            room_depth = self.pars['widths'][batch]
            batch_history = []
            self.state = [round(random.uniform(-room_width/2, room_width/2), 2),
                          round(random.uniform(-room_depth/2, room_depth/2), 2)]

            for step in range(self.pars['t_episode']):
                # action = actions[batch, :, step] / np.linalg.norm(actions[batch, :, step])
                # Generate action from given policy
                action, direc = policy(observation)
                actions[batch, :, step] = action
                direcs[batch, :, step] = direc

                # Determine state transitioned to
                new_state = self.state + [self.pars['agent_step_size'] * i for i in action]
                new_state = np.array([np.clip(new_state[0], a_min=-room_width / 2, a_max=room_width / 2),
                                      np.clip(new_state[1], a_min=-room_depth / 2, a_max=room_depth / 2)])
                reward = 0  # If you get reward, it should be coded here
                transition = {"action": action, "state": self.state, "next_state": new_state,
                              "reward": reward, "step": self.global_steps}
                batch_history.append(transition)
                self.state = new_state
                observation = self.make_observation()

                observations[batch, :, step] = observation
                new_states[batch, :, step] = new_state
                rewards.append(reward)
            self.history.append(batch_history)

        return observations, new_states, rewards, actions, direcs

    def plot_trajectory(self, history_data=None, ax=None):
        if history_data is None:
            history_data = self.history
            # history_data = []
            # for i in range(0, 400, 25):
            #     history_data.append(self.history[i:i+25])
        if ax is None:
            mlp.rc('font', size=6)
            fig = plt.figure(figsize=(8, 6))

        for batch in range(16):
            ax = plt.subplot(4, 4, batch + 1)
            room_width = self.pars['widths'][batch]
            room_depth = room_width

            ax.plot([-room_width / 2, room_width / 2],
                    [-room_depth / 2, -room_depth / 2], "r", lw=2)
            ax.plot([-room_width / 2, room_width / 2],
                    [room_depth / 2, room_depth / 2], "r", lw=2)
            ax.plot([-room_width / 2, -room_width / 2],
                    [-room_depth / 2, room_depth / 2], "r", lw=2)
            ax.plot([room_width / 2, room_width / 2],
                    [-room_depth / 2, room_depth / 2], "r", lw=2)

            state_history = [s["state"] for s in history_data[batch]]
            next_state_history = [s["next_state"] for s in history_data[batch]]

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
            ax.set_xlim([-6, 6])
            ax.set_xticks([-5, 0, 5])
            ax.set_ylim([-6, 6])
            ax.set_yticks([-5, 0, 5])

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(sc, cax=cbar_ax)
        cbar_ax.set_ylabel('N steps', rotation=270)

        return ax
