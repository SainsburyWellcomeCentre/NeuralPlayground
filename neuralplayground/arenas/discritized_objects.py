import random

import matplotlib.pyplot as plt
import numpy as np

from neuralplayground.arenas.arena_core import Environment
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
        else:
            self.state_dims_labels = ["x_pos", "y_pos"]
            self.arena_x_limits = env_kwargs["arena_x_limits"]
            self.arena_y_limits = env_kwargs["arena_y_limits"]

        self.n_objects = env_kwargs["n_objects"]
        self.state_density = env_kwargs["state_density"]
        self.arena_limits = np.array(
            [[self.arena_x_limits[0], self.arena_x_limits[1]], [self.arena_y_limits[0], self.arena_y_limits[1]]]
        )
        self.room_width = np.diff(self.arena_x_limits)[0]
        self.room_depth = np.diff(self.arena_y_limits)[0]
        self.agent_step_size = env_kwargs["agent_step_size"]
        self._create_default_walls()
        self._create_custom_walls()
        self.wall_list = self.default_walls + self.custom_walls

        # Variables for discretised state space
        self.resolution_w = int(self.state_density * self.room_width)
        self.resolution_d = int(self.state_density * self.room_depth)
        self.x_array = np.linspace(
            -self.room_width / 2 + (1 / 2 * self.state_density),
            self.room_width / 2 - (1 / 2 * self.state_density),
            num=self.resolution_w,
        )
        self.y_array = np.linspace(
            -self.room_depth / 2 + (1 / 2 * self.state_density),
            self.room_depth / 2 - (1 / 2 * self.state_density),
            num=self.resolution_d,
        )
        self.mesh = np.array(np.meshgrid(self.x_array, self.y_array))
        self.xy_combination = np.stack(np.array(np.meshgrid(self.x_array, self.y_array)), axis=-1)
        self.ws = int(self.room_width * self.state_density)
        self.hs = int(self.room_depth * self.state_density)
        self.n_states = self.resolution_w * self.resolution_d
        self.objects = np.empty(shape=(self.n_states, self.n_objects))

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

        # Reset to first position recorded in this session
        if self.use_behavioral_data:
            pos, head_dir = self.experiment.position[0, :], self.experiment.head_direction[0, :]
            custom_state = np.concatenate([pos, head_dir])

        self.objects = self.generate_objects()

        # Fully observable environment, make_observation returns the state
        observation = self.make_object_observation(pos)
        self.state = observation
        return observation, self.state

    def step(self, action: np.ndarray, normalize_step: bool = False, skip_every: int = 10):
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
        else:
            if action[0] == 0:
                action_rev = np.array([0.0, -action[1]])
            else:
                action_rev = action
            if normalize_step and np.linalg.norm(action) > 0:
                action_rev = action_rev / np.linalg.norm(action_rev)
                new_pos_state = self.state[-1] + self.agent_step_size * action_rev
            else:
                new_pos_state = self.state[-1] + action_rev
            new_pos_state, valid_action = self.validate_action(self.state[-1], action_rev, new_pos_state)
        reward = self.reward_function(action, self.state[-1])  # If you get reward, it should be coded here
        observation = self.make_object_observation(new_pos_state)
        self.state = observation
        self.transition = {
            "action": action,
            "state": self.old_state,
            "next_state": self.state,
            "reward": reward,
            "step": self.global_steps,
        }
        # self.history.append(transition)
        self._increase_global_step()
        return observation, self.state

    def generate_objects(self):
        """
        Generate objects in the environment. In this case, the objects are one-hot encoded vectors.
        Returns
        -------
            objects: ndarray (n_states, n_objects)
                Array of the objects in the environment, one-hot encoded
        """
        poss_objects = np.zeros(shape=(self.n_objects, self.n_objects))
        for i in range(self.n_objects):
            for j in range(self.n_objects):
                if j == i:
                    poss_objects[i][j] = 1
        # Generate landscape of objects in each environment
        objects = np.zeros(shape=(self.n_states, self.n_objects))
        for i in range(self.n_states):
            rand = random.randint(0, self.n_objects - 1)
            objects[i, :] = poss_objects[rand]
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
        """
        Convert an (x,y) position to a discretised state index
        Parameters
        ----------
            pos: ndarray (2,)
                Vector of the x and y coordinate of the position of the animal in the environment
        Returns
        -------
            index: int
                Index of the state in the discretised state space
        """
        if len(pos) > 2:
            pos = pos[:2]
        diff = (self.xy_combination - pos) ** 2
        dist = np.sum(diff**2, axis=-1)
        index = np.argmin(dist)
        return index

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
            np.array([[self.arena_limits[0, 0], self.arena_limits[1, 0]], [self.arena_limits[0, 0], self.arena_limits[1, 1]]])
        )
        self.default_walls.append(
            np.array([[self.arena_limits[0, 1], self.arena_limits[1, 0]], [self.arena_limits[0, 1], self.arena_limits[1, 1]]])
        )
        self.default_walls.append(
            np.array([[self.arena_limits[0, 0], self.arena_limits[1, 0]], [self.arena_limits[0, 1], self.arena_limits[1, 0]]])
        )
        self.default_walls.append(
            np.array([[self.arena_limits[0, 0], self.arena_limits[1, 1]], [self.arena_limits[0, 1], self.arena_limits[1, 1]]])
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
            new_state, crossed = check_crossing_wall(pre_state=pre_state, new_state=new_state, wall=wall)
            crossed_wall = crossed or crossed_wall
        return new_state, crossed_wall

    # to be written again here
    def plot_objects(self, history_data=None, ax=None, return_figure=False):
        """Plot the Trajectory of the agent in the environment

        Parameters
        ----------
        history_data: None
            default to access to the saved history of positions in the environment
        ax: None
            default to create ax
        Returns
        -------
        Returns a plot of the trajectory of the animal in the environment
        """
        if history_data is None:
            history_data = self.history
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(8, 6))

        for wall in self.default_walls:
            ax.plot(wall[:, 0], wall[:, 1], "C3", lw=3)

        for wall in self.custom_walls:
            ax.plot(wall[:, 0], wall[:, 1], "C0", lw=3)

        if return_figure:
            return f, ax
        else:
            return ax
