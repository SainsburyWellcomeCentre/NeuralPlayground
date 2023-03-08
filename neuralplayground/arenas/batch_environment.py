import matplotlib as mpl
import matplotlib.pyplot as plt
from neuralplayground.arenas.arena_core import Environment
from neuralplayground.arenas.simple2d import Simple2D
import numpy as np
from neuralplayground.utils import check_crossing_wall


class BatchEnvironment(Environment):
    def __init__(self, environment_name: str = "BatchEnv", env_class: object = Simple2D, batch_size: int = 1,
                 **env_kwargs):
        super().__init__(environment_name, **env_kwargs)
        self.batch_size = batch_size
        batch_x_limits = env_kwargs['arena_x_limits']
        batch_y_limits = env_kwargs['arena_y_limits']
        self.environments = []
        for i in range(self.batch_size):
            env_kwargs['arena_x_limits'] = batch_x_limits[i]
            env_kwargs['arena_y_limits'] = batch_y_limits[i]
            self.environments.append(env_class(**env_kwargs))

        self.room_widths = [np.diff(self.environments[i].arena_x_limits)[0] for i in range(self.batch_size)]
        self.room_depths = [np.diff(self.environments[i].arena_y_limits)[0] for i in range(self.batch_size)]
        self.state_densities = [self.environments[i].state_density for i in range(self.batch_size)]

    def reset(self, random_state: bool = False, custom_state: np.ndarray = None):
        self.global_steps = 0
        self.global_time = 0
        self.history = []

        all_observations = []
        all_states = []
        for i, env in enumerate(self.environments):
            env_obs, env_state = env.reset(random_state=random_state, custom_state=custom_state)
            all_states.append(env_state)
            all_observations.append(env_obs)

        return all_observations, all_states

    def step(self, actions: np.ndarray, normalize_step: bool = False):
        all_observations = []
        all_states = []
        all_rewards = []
        all_allowed = True
        for batch, env in enumerate(self.environments):
            action = actions[batch]
            env_obs, env_state = env.step(action, normalize_step)
            if env.state[0] == env.old_state[0]:
                all_allowed = False
            all_observations.append(env_obs)
            all_states.append(env_state)

        if not all_allowed:
            for env in self.environments:
                env.state = env.old_state
        else:
            self.history.append([env.transition for env in self.environments])

        return all_observations, all_states

    def plot_trajectory(self, history_data: list = None, ax=None, return_figure: bool = False, save_path: str = None,
                        plot_every: int = 1):
        """ Plot the Trajectory of the agent in the environment
        Parameters
        ----------
        history_data: list of interactions
            if None, use history data saved as attribute of the arena, use custom otherwise
        ax: mpl.axes._subplots.AxesSubplot (matplotlib axis from subplots)
            axis from subplot from matplotlib where the trajectory will be plotted.
        return_figure: bool
            If true, it will return the figure variable generated to make the plot
        save_path: str, list of str, tuple of str
            saving path of the generated figure, if None, no figure is saved
        Returns
        -------
        ax: mpl.axes._subplots.AxesSubplot (matplotlib axis from subplots)
            Modified axis where the trajectory is plotted
        f: matplotlib.figure
            if return_figure parameters is True
        """
        env = self.environments[3]
        # Use or not saved history
        if history_data is None:
            history_data = [his[3] for his in self.history]

        # Generate Figure
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Draw walls
        for wall in env.default_walls:
            ax.plot(wall[:, 0], wall[:, 1], "C3", lw=3)

        # Draw custom walls
        for wall in env.custom_walls:
            ax.plot(wall[:, 0], wall[:, 1], "C0", lw=3)

        # Making the trajectory plot roughly square to show structure of the arena better
        lower_lim, upper_lim = np.amin(env.arena_limits), np.amax(env.arena_limits)
        ax.set_xlim([lower_lim, upper_lim])
        ax.set_ylim([lower_lim, upper_lim])

        # Make plot of positions
        if len(history_data) != 0:
            state_history = [s["state"][-1] for s in history_data]
            next_state_history = [s["next_state"][-1] for s in history_data]
            starting_point = state_history[0]
            ending_point = next_state_history[-1]

            cmap = mpl.cm.get_cmap("plasma")
            norm = plt.Normalize(0, len(state_history))

            aux_x = []
            aux_y = []
            for i, s in enumerate(state_history):

                if i % plot_every == 0:
                    if i + plot_every >= len(state_history):
                        break
                    x_ = [s[0], state_history[i+plot_every][0]]
                    y_ = [s[1], state_history[i+plot_every][1]]
                    aux_x.append(s[0])
                    aux_y.append(s[1])
                    sc = ax.plot(x_, y_, "-", color=cmap(norm(i)), alpha=0.6)

            sc = ax.scatter(aux_x, aux_y, c=np.arange(len(aux_x)), vmin=0, vmax=len(aux_x),
                            cmap="plasma", alpha=0.6, s=0.1)
            cbar = plt.colorbar(sc, ax=ax, ticks=[0, len(state_history)])
            cbar.ax.set_ylabel('N steps', rotation=270, fontsize=16)
            cbar.ax.set_yticklabels([0, len(state_history)], fontsize=16)

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")

        if return_figure:
            return ax, f
        else:
            return ax
