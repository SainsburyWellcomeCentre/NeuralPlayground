import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random

from .simple2d import Simple2D

class DiscreteObjectEnvironment(Simple2D):
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
    def __init__(self, environment_name='DiscreteObject', **env_kwargs):
        self.number_object= env_kwargs['number_object']
        self.state_density = env_kwargs['state_density']
        self.arena_x_limits = env_kwargs['arena_x_limits']
        self.arena_y_limits = env_kwargs['arena_y_limits']
        self.room_width = np.diff(self.arena_x_limits)[0]
        self.room_depth = np.diff(self.arena_y_limits)[0]

        # Variables for discretised state space
        self.n_states = (self.room_width * self.room_depth) * self.state_density
        self.resolution_w = int(self.state_density * self.room_width)
        self.resolution_d = int(self.state_density * self.room_depth)
        self.x_array = np.linspace(-self.room_width / 2 + 0.5, self.room_width / 2 - 0.5, num=self.resolution_w)
        self.y_array = np.linspace(self.room_depth / 2 - 0.5, -self.room_depth / 2 + 0.5, num=self.resolution_d)
        self.mesh = np.array(np.meshgrid(self.x_array, self.y_array))
        self.xy_combination = np.array(np.meshgrid(self.x_array, self.y_array)).T
        self.ws = int(self.room_width * self.state_density)
        self.hs = int(self.room_depth * self.state_density)
        super().__init__(environment_name, **env_kwargs)

    def reset(self, random_state=False, custom_state=False):
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
        if random_state:
            pos_state = [np.random.uniform(low=self.arena_limits[0, 0], high=self.arena_limits[0, 1]),
                          np.random.uniform(low=self.arena_limits[1, 0], high=self.arena_limits[1, 1])]
        else:
            pos_state = [0, 0]
        pos_state = np.array(pos_state)

        if custom_state is not None:
            pos_state = np.array(custom_state)

        self.objects = np.zeros(shape=(self.n_states, self.number_object))
        self.generate_objects()

        # Fully observable environment, make_observation returns the state
        observation = self.make_observation(pos_state)
        self.state = observation
        return observation, self.state

    def step(self, action, normalize_step=False):
        """
        Runs the environment dynamics. Increasing global counters. Given some action, return observation,
        new state and reward.

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
            Array of the observation of the agent in the environment, in this case the sensory object.
        """
        if normalize_step:
            action = action/np.linalg.norm(action)
            new_pos_state = self.state[-1] + self.agent_step_size*action
        else:
            new_pos_state = self.state[-1] + action
        new_pos_state, valid_action = self.validate_action(self.state[-1], action, new_pos_state)
        reward = self.reward_function(action, self.state)  # If you get reward, it should be coded here
        observation = self.make_observation(new_pos_state)
        new_state = observation
        self.state = new_state
        transition = {"action": action, "state": self.state, "next_state": new_state,
                      "reward": reward, "step": self.global_steps}
        self.history.append(transition)
        self._increase_global_step()
        return observation, new_state, reward

    def generate_objects(self):
        poss_objects = np.zeros(shape=(self.number_object,self.number_object))
        for i in range(self.number_object):
            for j in range(self.number_object):
                if j == i:
                    poss_objects[i][j] = 1
        # Generate landscape of objects in each environment
        for i in range(self.n_states):
            rand = random.randint(0, self.number_object - 1)
            self.objects[i, :] = poss_objects[rand]

    def make_observation(self, pos):
        index = self.pos_to_state(pos)
        object = self.objects[index]
        return [index, object, pos]

    def pos_to_state(self, pos):
        diff = self.xy_combination - pos[np.newaxis, ...]
        dist = np.sum(diff ** 2, axis=2).T
        index = np.argmin(dist)
        return index

    #to be written again here
    def plot_objects(self, history_data=None, ax=None, return_figure=False):
        """ Plot the Trajectory of the agent in the environment

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
