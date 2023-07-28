import cv2
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.spaces import Box
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from neuralplayground.arenas.arena_core import Environment
from neuralplayground.plotting.plot_utils import make_plot_trajectories
from neuralplayground.utils import check_crossing_wall


class Simple2D(Environment):
    """
    Methods (Some in addition to Environment class)
    ----------
    __init__(self, environment_name="2DEnv", **env_kwargs):
        Initialise the class
    reset(self):
        Reset the environment variables
    step(self, action):
        Increment the global step count of the agent in the environment and moves
        the agent in a random direction with a fixed step size
    plot_trajectory(self, history_data=None, ax=None):
        Plot the Trajectory of the agent in the environment. In addition to environment class.
    _create_default_walls(self):
        Generates outer border of the 2D environment based on the arena limits

    Attributes (Some in addition to the Environment class)
    ----------
    state: ndarray
        Contains the x, y coordinate of the position and head direction of the agent (will be further developed)
        head_direction: ndarray
                Contains the x and y Coordinates of the position
        position: ndarray
                Contains the x and y Coordinates of the position
    history: list of dicts
        Saved history over simulation steps (action, state, new_state, reward, global_steps)
    global_steps: int
        Counter of the number of steps in the environment
    room_width: int
        Size of the environment in the x direction
    room_depth: int
        Size of the environment in the y direction
    metadata: dict
        Dictionary containing the metadata
    observation: ndarray
        Fully observable environment, make_observation returns the state
        Array of the observation of the agent in the environment (Could be modified as the environments are evolves)
    agent_step_size: float
         Size of the step when executing movement, agent_step_size*global_steps will give
         a measure of the total distance traversed by the agent
    """

    def __init__(self, environment_name: str = "2DEnv", **env_kwargs):
        """Initialise the class

        Parameters
        ----------
        env_kwargs: dict
        Dictionary with parameters of the experiment of the children class
            time_step_size: float
                time_step_size * global_steps will give a measure of the time in the experimental setting
            agent_step_size: float
                Step size used when the action is a direction in x,y coordinate (normalize false in step())
                Agent_step_size * global_step_number will give a measure of the distance in the experimental setting
            arena_x_limits: float
                Size of the environment in the x direction
            arena_y_limits: float
                Size of the environment in the y direction

        environment_name: str
            Name of the specific instantiation of the Simple2D class

        """
        super().__init__(environment_name, **env_kwargs)
        self.metadata = {"env_kwargs": env_kwargs}
        self.arena_x_limits, self.arena_y_limits = (
            env_kwargs["arena_x_limits"],
            env_kwargs["arena_y_limits"],
        )
        self.arena_limits = np.array(
            [
                [self.arena_x_limits[0], self.arena_x_limits[1]],
                [self.arena_y_limits[0], self.arena_y_limits[1]],
            ]
        )
        self.room_width = np.diff(self.arena_x_limits)[0]
        self.room_depth = np.diff(self.arena_y_limits)[0]
        self.observation_space = Box(
            low=np.array([self.arena_x_limits[0], self.arena_y_limits[0]]),
            high=np.array([self.arena_x_limits[1], self.arena_y_limits[1]]),
            dtype=np.float64,
        )
        # Action are not bounded in this environment since the action is normalized in the step function
        # And can't cross the walls of the environment
        self.action_space = Box(
            low=np.array((-np.inf, -np.inf)),
            high=np.array((np.inf, np.inf)),
            dtype=np.float64,
        )
        self.agent_step_size = env_kwargs["agent_step_size"]
        self.state_dims_labels = ["x_pos", "y_pos"]

        self._create_default_walls()
        self._create_custom_walls()
        self.wall_list = self.default_walls + self.custom_walls
        self.state_dims_labels = ["x_pos", "y_pos"]
        self.reset()

    def _create_default_walls(self):
        """Generate walls to limit the arena based on the limits given in kwargs when initializing the object.
        Each wall is presented by a matrix
            [[xi, yi],
             [xf, yf]]
        where xi and yi are x y coordinates of one limit of the wall, and xf and yf are coordinates of the other limit.
        Walls are added to default_walls list, to then merge it with custom ones.
        See notebook with custom arena examples.
        """
        self.default_walls = []
        self.default_walls.append(
            np.array(
                [
                    [self.arena_limits[0, 0], self.arena_limits[1, 0]],
                    [self.arena_limits[0, 0], self.arena_limits[1, 1]],
                ]
            )
        )
        self.default_walls.append(
            np.array(
                [
                    [self.arena_limits[0, 1], self.arena_limits[1, 0]],
                    [self.arena_limits[0, 1], self.arena_limits[1, 1]],
                ]
            )
        )
        self.default_walls.append(
            np.array(
                [
                    [self.arena_limits[0, 0], self.arena_limits[1, 0]],
                    [self.arena_limits[0, 1], self.arena_limits[1, 0]],
                ]
            )
        )
        self.default_walls.append(
            np.array(
                [
                    [self.arena_limits[0, 0], self.arena_limits[1, 1]],
                    [self.arena_limits[0, 1], self.arena_limits[1, 1]],
                ]
            )
        )

    def _create_custom_walls(self):
        """Custom walls method. In this case is empty since the environment is a simple square room
        Override this method to generate more walls, see jupyter notebook with examples
        """
        self.custom_walls = []

    def reset(self, random_state: bool = False, custom_state: np.ndarray = None):
        """Reset the environment variables

        Parameters
        ----------
        random_state: bool
            If True, sample a new position uniformly within the arena, use default otherwise
        custom_state: np.ndarray
            If given, use this array to set the initial state

        Returns
        ----------
        observation: ndarray
            Because this is a fully observable environment, make_observation returns the state of the environment
            Array of the observation of the agent in the environment (Could be modified as the environments are evolves)

        self.state: ndarray (2,)
            Vector of the x and y coordinate of the position of the animal in the environment
        """
        self.global_steps = 0
        self.global_time = 0
        self.history = []
        if random_state:
            self.state = [
                np.random.uniform(low=self.arena_limits[0, 0], high=self.arena_limits[0, 1]),
                np.random.uniform(low=self.arena_limits[1, 0], high=self.arena_limits[1, 1]),
            ]
        else:
            self.state = [0, 0]
        self.state = np.array(self.state)

        if custom_state is not None:
            self.state = np.array(custom_state)
        # Fully observable environment, make_observation returns the state
        observation = self.make_observation()
        return observation, self.state

    def step(self, action: np.ndarray, normalize_step: bool = False):
        """Runs the environment dynamics. Increasing global counters.
        Given some action, return observation, new state and reward.

        Parameters
        ----------
        action: ndarray (2,)
            Array containing the action of the agent, in this case the delta_x and detla_y increment to position
        normalize_step: bool
            If true, the action is normalized to have unit size, then scaled by the agent step size

        Returns
        -------
        reward: float
            The reward that the animal receives in this state
        new_state: ndarray
            Update the state with the updated vector of coordinate x and y of position and head directions respectively
        observation: ndarray
            Array of the observation of the agent in the environment
        """
        if normalize_step:
            action = action / np.linalg.norm(action)
            new_state = self.state + self.agent_step_size * action
        else:
            new_state = self.state + action
        new_state, valid_action = self.validate_action(self.state, action, new_state)
        reward = self.reward_function(action, self.state)  # If you get reward, it should be coded here
        transition = {
            "action": action,
            "state": self.state,
            "next_state": new_state,
            "reward": reward,
            "step": self.global_steps,
        }
        self.history.append(transition)
        self.state = new_state
        observation = self.make_observation()
        self._increase_global_step()
        return observation, new_state, reward

    def validate_action(self, pre_state, action, new_state):
        """Check if the new state is crossing any walls in the arena.

        Parameters
        ----------
        pre_state : (2,) 2d-ndarray
            2d position of pre-movement
        new_state : (2,) 2d-ndarray
            2d position of post-movement

        Returns
        -------
        new_state: (2,) 2d-ndarray
            corrected new state. If it is not crossing the wall, then the new_state stays the same, if the state cross the
            wall, new_state will be corrected to a valid place without crossing the wall
        crossed_wall: bool
            True if the change in state crossed a wall and was corrected
        """
        crossed_wall = False
        for wall in self.wall_list:
            new_state, crossed = check_crossing_wall(pre_state=pre_state, new_state=new_state, wall=wall)
            crossed_wall = crossed or crossed_wall
        return new_state, crossed_wall

    def plot_trajectory(
        self,
        history_data: list = None,
        ax=None,
        return_figure: bool = False,
        save_path: str = None,
        plot_every: int = 1,
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
        # Use or not saved history
        if history_data is None:
            history_data = self.history

        # Generate Figure
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Make plot of positions
        if len(history_data) != 0:
            state_history = [s["state"] for s in history_data]
            x = []
            y = []
            for i, s in enumerate(state_history):
                # if i % plot_every == 0:
                #     if i + plot_every >= len(state_history):
                #         break
                x.append(s[0])
                y.append(s[1])
            ax = make_plot_trajectories(self.arena_limits, np.asarray(x), np.asarray(y), ax, plot_every)

        for wall in self.default_walls:
            ax.plot(wall[:, 0], wall[:, 1], "C3", lw=3)

            # Draw custom walls
        for wall in self.custom_walls:
            ax.plot(wall[:, 0], wall[:, 1], "C0", lw=3)

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")

        if return_figure:
            return ax, f
        else:
            return ax

    def render(self, history_length=30):
        """Render the environment live through iterations"""
        f, ax = plt.subplots(1, 1, figsize=(8, 6))
        canvas = FigureCanvas(f)
        history = self.history[-history_length:]
        ax = self.plot_trajectory(history_data=history, ax=ax)
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(f.canvas.get_width_height()[::-1] + (3,))
        print(image.shape)
        cv2.imshow("2D_env", image)
        cv2.waitKey(10)
