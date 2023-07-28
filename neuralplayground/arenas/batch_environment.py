import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from neuralplayground.arenas.arena_core import Environment
from neuralplayground.arenas.simple2d import Simple2D


class BatchEnvironment(Environment):
    def __init__(self, environment_name: str = "BatchEnv", env_class: object = Simple2D, batch_size: int = 1, **env_kwargs):
        """
        Initialise a batch of environments. This is useful for training a single agent on multiple environments simultaneously.
        Parameters
        ----------
            environment_name: str
                Name of the environment
            env_class: object
                Class of the environment
            batch_size: int
                Number of environments in the batch
            **env_kwargs: dict
                Keyword arguments for the environment
        """
        super().__init__(environment_name, **env_kwargs)
        self.batch_size = batch_size
        self.batch_x_limits = env_kwargs["arena_x_limits"]
        self.batch_y_limits = env_kwargs["arena_y_limits"]
        self.use_behavioural_data = env_kwargs["use_behavioural_data"]
        self.environments = []
        for i in range(self.batch_size):
            env_kwargs["arena_x_limits"] = self.batch_x_limits[i]
            env_kwargs["arena_y_limits"] = self.batch_y_limits[i]
            self.environments.append(env_class(**env_kwargs))

        self.room_widths = [np.diff(self.environments[i].arena_x_limits)[0] for i in range(self.batch_size)]
        self.room_depths = [np.diff(self.environments[i].arena_y_limits)[0] for i in range(self.batch_size)]
        self.state_densities = [self.environments[i].state_density for i in range(self.batch_size)]

    def reset(self, random_state: bool = True, custom_state: np.ndarray = None):
        """
        Reset the environment
        Parameters
        ----------
            random_state: bool
                If True, the agent will be placed in a random state
            custom_state: np.ndarray
                If not None, the agent will be placed in the state specified by custom_state
        Returns
        -------
            all_observations: list of np.ndarray
                List of observations for each environment in the batch
            all_states: list of np.ndarray
                List of states for each environment in the batch
        """
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
        """
        Step the environment
        Parameters
        ----------
            actions: np.ndarray
                Array of actions for each environment in the batch
            normalize_step: bool
                If True, the agent will be placed in the state specified by custom_state
        Returns
        -------
            all_observations: list of np.ndarray
                List of observations for each environment in the batch
            all_states: list of np.ndarray
                List of states for each environment in the batch
        """
        all_observations = []
        all_states = []
        all_allowed = True
        for batch, env in enumerate(self.environments):
            action = actions[batch]
            env_obs, env_state = env.step(action, normalize_step)
            if self.use_behavioural_data:
                if env.state[0] == env.old_state[0]:
                    all_allowed = False
            else:
                if env.state[0] == env.old_state[0] and action != [0, 0]:
                    all_allowed = False
            all_observations.append(env_obs)
            all_states.append(env_state)

        if not all_allowed:
            for env in self.environments:
                env.state = env.old_state
        else:
            self.history.append([env.transition for env in self.environments])

        return all_observations, all_states

    def plot_trajectory(
        self, history_data: list = None, ax=None, return_figure: bool = False, save_path: str = None, plot_every: int = 1
    ):
        """Plot the Trajectory of the agent in the environment
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
        env = self.environments[0]
        # Use or not saved history
        if history_data is None:
            history_data = [his[0] for his in self.history]

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
            state_history[0]
            next_state_history[-1]

            cmap = mpl.cm.get_cmap("plasma")
            norm = plt.Normalize(0, len(state_history))

            aux_x = []
            aux_y = []
            for i, s in enumerate(state_history):
                if i % plot_every == 0:
                    if i + plot_every >= len(state_history):
                        break
                    x_ = [s[0], state_history[i + plot_every][0]]
                    y_ = [s[1], state_history[i + plot_every][1]]
                    aux_x.append(s[0])
                    aux_y.append(s[1])
                    sc = ax.plot(x_, y_, "-", color=cmap(norm(i)), alpha=0.6)

            sc = ax.scatter(aux_x, aux_y, c=np.arange(len(aux_x)), vmin=0, vmax=len(aux_x), cmap="plasma", alpha=0.6, s=0.1)
            cbar = plt.colorbar(sc, ax=ax, ticks=[0, len(state_history)])
            cbar.ax.set_ylabel("N steps", rotation=270, fontsize=16)
            cbar.ax.set_yticklabels([0, len(state_history)], fontsize=16)

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")

        if return_figure:
            return ax, f
        else:
            return ax

    def collect_environment_info(self, model_input, history, environments):
        """
        Collect information about the environment for each step of the trajectory.
        Parameters
        ----------
            model_input: list of np.ndarray
                List of model inputs for each step of the trajectory
            history: list of np.ndarray
                List of histories for each step of the trajectory
            environments: list of dict
                List of environments for each step of the trajectory
        Returns
        -------
            environments: list of dict
                List of environments for each step of the trajectory
        """
        for step in range(len(model_input)):
            id = model_input[step][0][0]["id"]
            if not any(d["id"] == id for d in environments[0]):
                x, y = history[step][0][-1][0], history[step][0][-1][1]

                # Round the (x, y) coordinates to the center of the nearest state
                rounded_x, rounded_y = self.round_to_nearest_state_center(x, y)

                # Normalize the rounded coordinates
                normalized_x, normalized_y = self.normalize_coordinates(rounded_x, rounded_y)

                loc_dict = {
                    "id": id,
                    "observation": int(np.argmax(model_input[step][1])),
                    "x": normalized_x,
                    "y": normalized_y,
                    "shiny": None,
                }
                environments[0].append(loc_dict)

        environments[0] = sorted(environments[0], key=lambda x: x["id"])

        return environments

    def round_to_nearest_state_center(self, x, y):
        """
        Round the (x, y) coordinates to the center of the nearest state.
        Parameters
        ----------
            x: float
                x coordinate
            y: float
                y coordinate
        Returns
        -------
            rounded_x: float
                x coordinate rounded to the center of the nearest state
            rounded_y: float
                y coordinate rounded to the center of the nearest state
        """
        state_width = 1 / self.state_densities[0]
        state_depth = 1 / self.state_densities[1]

        half_state_width = state_width / 2
        half_state_depth = state_depth / 2

        rounded_x = round((x + half_state_width) / state_width) * state_width - half_state_width
        rounded_y = round((y + half_state_depth) / state_depth) * state_depth - half_state_depth

        return rounded_x, rounded_y

    def normalize_coordinates(self, x, y):
        """
        Normalize the (x, y) coordinates to the range [0, 1].
        Parameters
        ----------
            x: float
                x coordinate
            y: float
                y coordinate
        Returns
        -------
            normalized_x: float
                x coordinate normalized to the range [0, 1]
            normalized_y: float
                y coordinate normalized to the range [0, 1]
        """
        x_min, x_max = self.batch_x_limits[0][0], self.batch_x_limits[0][1]
        y_min, y_max = self.batch_y_limits[0][0], self.batch_y_limits[0][1]
        normalized_x = (x - x_min) / (x_max - x_min)
        normalized_y = (y - y_min) / (y_max - y_min)
        return normalized_x, normalized_y
