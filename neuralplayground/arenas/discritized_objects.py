import random

import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from neuralplayground.arenas.arena_core import Environment
from neuralplayground.plotting.plot_utils import make_plot_trajectories
from neuralplayground.utils import check_crossing_wall


class DiscreteObjectEnvironment(Environment):
    """
    Arena class which accounts for discrete sensory objects, inherits from the Simple2D class.

    Methods
    ------
        __init__(self, environment_name='DiscreteObject', **env_kwargs):
            Initialize the class. env_kwargs arguments are specific for each of the child environments and
            described in their respective class.
        reset(self):
            Re-initialize state and global counters. Resets discrete objects at each state.
        generate_objects(self):
            Randomly distribute objects (one-hot encoded vectors) at each discrete state within the environment
        make_observation(self, step):
            Convert an (x,y) position into an observation of an object
        pos_to_state(self, step):
            Convert an (x,y) position to a discretised state index
        plot_objects(self, history_data=None, ax=None, return_figure=False):

    Attributes
        ----------
        state: array
            Empty array for this abstract class
        history: list
            Contains transition history
        env_kwags: dict
            Arguments given to the init method
        global_steps: int
            Number of calls to step method, set to 0 when calling reset
        global_time: float
            Time simulating environment through step method, then global_time = time_step_size * global_steps
        number_object: int
            The number of possible objects present at any state
        room_width: int
            Size of the environment in the x direction
        room_depth: int
            Size of the environment in the y direction
        state_density: int
            The density of discrete states in the environment
    """

    def __init__(
        self,
        recording_index: int = None,
        environment_name: str = "DiscreteObject",
        verbose: bool = False,
        experiment_class: str = None,
        **env_kwargs,
    ):
        """
        Initialize the class. env_kwargs arguments are specific for each of the child environments and
        described in their respective class.
        Parameters
        ----------
            environment_name: str
                Name of the environment
            verbose: bool
                If True, print information about the environment
            experiment_class: str
                Name of the class of the experiment to use
            **env_kwargs:
                Arguments specific to each environment
        """
        super().__init__(environment_name, **env_kwargs)
        self.environment_name = environment_name
        self.use_behavioral_data = env_kwargs["use_behavioural_data"]
        self.experiment = experiment_class(
            experiment_name=self.environment_name,
            data_path=env_kwargs["data_path"],
            recording_index=recording_index,
            verbose=verbose,
        )
        if self.use_behavioral_data:
            self.state_dims_labels = ["x_pos", "y_pos", "head_direction_x", "head_direction_y"]
            self.arena_limits = self.experiment.arena_limits
            self.arena_x_limits = self.arena_limits[0].astype(int)
            self.arena_y_limits = self.arena_limits[1].astype(int)
            self.state_density = 0.25
        else:
            self.state_dims_labels = ["x_pos", "y_pos"]
            self.arena_x_limits = env_kwargs["arena_x_limits"]
            self.arena_y_limits = env_kwargs["arena_y_limits"]
            self.state_density = env_kwargs["state_density"]

        self.n_objects = env_kwargs["n_objects"]
        self.arena_limits = np.array(
            [[self.arena_x_limits[0], self.arena_x_limits[1]], [self.arena_y_limits[0], self.arena_y_limits[1]]]
        )
        self.room_width = self.arena_x_limits[1] - self.arena_x_limits[0]
        self.room_depth = self.arena_y_limits[1] - self.arena_y_limits[0]
        self.agent_step_size = env_kwargs["agent_step_size"]
        self._create_default_walls()
        self._create_custom_walls()
        self.wall_list = self.default_walls + self.custom_walls

        # Variables for discretised state space
        self.resolution_w = int(self.room_width * self.state_density)
        self.resolution_d = int(self.room_depth * self.state_density)
        self.state_size = 1 / self.state_density

        self.x_array = np.linspace(self.arena_x_limits[0] + self.state_size/2, 
                                   self.arena_x_limits[1] - self.state_size/2, 
                                   self.resolution_w)
        self.y_array = np.linspace(self.arena_y_limits[0] + self.state_size/2, 
                                   self.arena_y_limits[1] - self.state_size/2, 
                                   self.resolution_d)
        self.mesh = np.meshgrid(self.x_array, self.y_array)
        self.xy_combination =  np.column_stack([self.mesh[0].ravel(), self.mesh[1].ravel()])
        self.n_states = self.resolution_w * self.resolution_d
        self.objects = self.generate_objects()
        self.occupancy_grid = np.zeros((self.resolution_d, self.resolution_w))

    def reset(self, random_state=True, custom_state=None):
        """
        Reset the environment variables and distribution of sensory objects.
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
        self.state = [-1, -1, [self.room_width + 1, self.room_depth + 1]]
        if random_state:
            pos = [
                np.random.uniform(low=self.arena_limits[0, 0], high=self.arena_limits[0, 1]),
                np.random.uniform(low=self.arena_limits[1, 0], high=self.arena_limits[1, 1]),
            ]
        else:
            pos = np.array([0, 0])

        if custom_state is not None:
            pos = np.array(custom_state)

        # Snap the initial position to the nearest discrete state
        x_index = np.clip(int((pos[0] - self.arena_x_limits[0]) // self.state_size), 0, self.resolution_w - 1)
        y_index = np.clip(int((pos[1] - self.arena_y_limits[0]) // self.state_size), 0, self.resolution_d - 1)
        pos = np.array([self.x_array[x_index], self.y_array[y_index]])

        # Reset to first position recorded in this session
        if self.use_behavioral_data:
            pos, head_dir = self.experiment.position[0, :], self.experiment.head_direction[0, :]
            custom_state = np.concatenate([pos, head_dir])

        self.objects = self.generate_objects()
        self.occupancy_grid = np.zeros((self.resolution_d, self.resolution_w))

        # Fully observable environment, make_observation returns the state
        observation = self.make_object_observation(pos)
        self.state = observation
        return observation, self.state

    def step(self, action: np.ndarray, normalize_step: bool = True, skip_every: int = 10):
        """
        Runs the environment dynamics. Increasing global counters. Given some action, return observation,
        new state and reward.

        Parameters
        ----------
        action: ndarray (2,)
            Array containing the action of the agent, in this case the delta_x and detla_y increment to position
        normalize_step: bool
            If true, the action is normalized to have unit size, then scaled by the agent step size
        skip_every: int
            When using behavioral data, the next state will be the position and head direction
            "skip_every" recorded steps after the current one, essentially reducing the sampling rate

        Returns
        -------
        reward: float
            The reward that the animal receives in this state
        new_state: ndarray
            Update the state with the updated vector of coordinate x and y of position and head directions respectively
        observation: ndarray
            Array of the observation of the agent in the environment, in this case the sensory object.
        """
        self.old_state = self.state.copy()
        if self.use_behavioral_data:
            # In this case, the action is ignored and computed from the step in behavioral data recorded from the experiment
            if self.global_steps * skip_every >= self.experiment.position.shape[0] - 1:
                self.global_steps = np.random.choice(np.arange(skip_every))
            # Time elapsed since last reset
            self.global_time = self.global_steps * self.time_step_size

            # New state as "skip every" steps after the current one in the recording
            new_pos_state = (
                self.experiment.position[self.global_steps * skip_every, :],
                self.experiment.head_direction[self.global_steps * skip_every, :],
            )
            new_pos_state = np.concatenate(new_pos_state)
        if not self.use_behavioral_data:
            action_rev = action
            if normalize_step:
                new_pos_state = np.add(self.state[-1], [self.agent_step_size * e for e in action_rev]).tolist()
            else:
                new_pos_state = np.add(self.state[-1], action_rev).tolist()
            new_pos_state, invalid_action = self.validate_action(
                self.state[-1], [self.agent_step_size * e for e in action_rev], new_pos_state[:2]
            )
        reward = self.reward_function(action, self.state[-1])  # If you get reward, it should be coded here
        observation = self.make_object_observation(new_pos_state)
        state_index = self.pos_to_state(new_pos_state)
        self.occupancy_grid[state_index // self.resolution_w, state_index % self.resolution_w] += 1
        self.state = observation
        self.transition = {
            "action": action,
            "state": self.old_state[-1],
            "next_state": self.state[-1],
            "reward": reward,
            "step": self.global_steps,
        }
        self.history.append(self.transition)
        self._increase_global_step()
        return observation, self.state, reward

    def generate_objects(self):
        """
        Randomly assign objects to each state.
        """
        n_states = self.resolution_w * self.resolution_d
        object_indices = np.random.randint(0, self.n_objects, size=n_states)
        objects = np.eye(self.n_objects)[object_indices]
        return objects


    def make_object_observation(self, pos):
        """
        Make an observation of the object in the environment at the current position.
        Parameters
        ----------
            pos: ndarray (2,)
                Vector of the x and y coordinate of the position of the animal in the environment
        Returns
        -------
            observation: ndarray (n_objects,)
                Array of the observation of the agent in the environment, in this case the sensory object.
        """
        index = self.pos_to_state(np.array(pos))
        object = self.objects[index]

        return [index, object, pos]

    def pos_to_state(self, pos):
        """Convert an (x,y) position to a discretised state index"""
        if not self.use_behavioral_data and np.shape(pos) == (2, 2):
            pos = pos[0]
        elif self.use_behavioral_data and len(pos) > 2:
            pos = pos[:2]
        
        x_index = np.floor((pos[0] - self.arena_x_limits[0]) / self.state_size).astype(int)
        y_index = np.floor((pos[1] - self.arena_y_limits[0]) / self.state_size).astype(int)
        
        # Ensure indices are within bounds
        x_index = np.clip(x_index, 0, self.resolution_w - 1)
        y_index = np.clip(y_index, 0, self.resolution_d - 1)
        
        return y_index * self.resolution_w + x_index

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
            np.array([[self.arena_limits[0, 0] - 0.1, self.arena_limits[1, 0] + 0.1], [self.arena_limits[0, 0] - 0.1, self.arena_limits[1, 1] + 0.1]])
        )
        self.default_walls.append(
            np.array([[self.arena_limits[0, 1] + 0.1, self.arena_limits[1, 0] - 0.1], [self.arena_limits[0, 1] + 0.1, self.arena_limits[1, 1] + 0.1]])
        )
        self.default_walls.append(
            np.array([[self.arena_limits[0, 0] - 0.1, self.arena_limits[1, 0] - 0.1], [self.arena_limits[0, 1] + 0.1, self.arena_limits[1, 0] - 0.1]])
        )
        self.default_walls.append(
            np.array([[self.arena_limits[0, 0] - 0.1, self.arena_limits[1, 1] + 0.1], [self.arena_limits[0, 1] + 0.1, self.arena_limits[1, 1] + 0.1]])
        )


    def _create_custom_walls(self):
        """Custom walls method. In this case is empty since the environment is a simple square room
        Override this method to generate more walls, see jupyter notebook with examples"""
        self.custom_walls = []

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
            new_state, crossed = check_crossing_wall(pre_state=pre_state, new_state=np.asarray(new_state), wall=wall)
            crossed_wall = crossed or crossed_wall
        
        # Snap the new_state back to the nearest discrete state
        x_index = np.argmin(np.abs(self.x_array - new_state[0]))
        y_index = np.argmin(np.abs(self.y_array - new_state[1]))
        new_state = np.array([self.x_array[x_index], self.y_array[y_index]])
        
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

    def visualize_environment(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Visualize discretization
        ax1.set_title("Environment Discretization")
        for x in np.arange(self.arena_x_limits[0], self.arena_x_limits[1] + self.state_size, self.state_size):
            ax1.axvline(x, color='gray', linestyle='-', linewidth=1)
        for y in np.arange(self.arena_y_limits[0], self.arena_y_limits[1] + self.state_size, self.state_size):
            ax1.axhline(y, color='gray', linestyle='-', linewidth=1)
        ax1.scatter(self.xy_combination[:, 0], self.xy_combination[:, 1], color='red', s=20, zorder=2)
        ax1.set_aspect('equal')
        ax1.set_xlim(self.arena_x_limits)
        ax1.set_ylim(self.arena_y_limits)
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")

        # Visualize object assignment
        ax2.set_title("Object Assignment" + f" (n_objects={self.n_objects})," + f" (n_states={self.n_states})," + f" (grid={self.resolution_w}x{self.resolution_d})")
        object_grid = np.argmax(self.objects, axis=1).reshape((self.resolution_d, self.resolution_w))
        im = ax2.imshow(object_grid, cmap='tab20', extent=[*self.arena_x_limits, *self.arena_y_limits], origin='lower')
        plt.colorbar(im, ax=ax2, label="Object ID")

        # Add text labels for object IDs and scatter plot for xy_combination
        for i in range(self.resolution_d):
            for j in range(self.resolution_w):
                ax2.text(self.x_array[j], self.y_array[i], str(object_grid[i, j]), 
                         ha='center', va='center', color='white', fontweight='bold')
        
        ax2.scatter(self.xy_combination[:, 0], self.xy_combination[:, 1], color='red', s=20, zorder=2)

        ax2.set_aspect('equal')
        ax2.set_xlim(self.arena_x_limits)
        ax2.set_ylim(self.arena_y_limits)
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")

        plt.tight_layout()
        plt.show()

        # Print statistics
        print(f"Arena dimensions: {self.room_width} x {self.room_depth}")
        print(f"State density: {self.state_density}")
        print(f"State size: {self.state_size} x {self.state_size}")
        print(f"Number of discrete states: {self.resolution_w * self.resolution_d}")
        print(f"Number of unique objects: {self.n_objects}")
        print(f"Grid dimensions: {self.resolution_w} x {self.resolution_d}")
        
        # Print object distribution
        unique, counts = np.unique(np.argmax(self.objects, axis=1), return_counts=True)
        print("\nObject distribution:")
        for obj, count in zip(unique, counts):
            print(f"Object {obj}: {count} states ({count/(self.resolution_w * self.resolution_d):.2%})")

    def visualize_occupancy(self, log_scale=True):
        fig, ax = plt.subplots(figsize=(12, 10))
        cmap = plt.cm.YlOrRd
        
        if log_scale:
            # Use log scale for better visualization of differences
            im = ax.imshow(self.occupancy_grid, cmap=cmap, norm=LogNorm(), extent=[*self.arena_x_limits, *self.arena_y_limits], origin='lower')
        else:
            im = ax.imshow(self.occupancy_grid, cmap=cmap, extent=[*self.arena_x_limits, *self.arena_y_limits], origin='lower')
        
        plt.colorbar(im, ax=ax, label='Number of visits (log scale)' if log_scale else 'Number of visits')
        
        ax.set_title('Agent Occupancy Heatmap')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Add grid lines
        for x in np.arange(self.arena_x_limits[0], self.arena_x_limits[1] + self.state_size, self.state_size):
            ax.axvline(x, color='gray', linestyle='--', linewidth=0.5)
        for y in np.arange(self.arena_y_limits[0], self.arena_y_limits[1] + self.state_size, self.state_size):
            ax.axhline(y, color='gray', linestyle='--', linewidth=0.5)
        
        # Add text annotations for each cell
        for i in range(self.resolution_d):
            for j in range(self.resolution_w):
                value = self.occupancy_grid[i, j]
                text_color = 'white' if value > np.mean(self.occupancy_grid) else 'black'
                ax.text(self.x_array[j], self.y_array[i], f'{int(value)}', 
                        ha='center', va='center', color=text_color, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
