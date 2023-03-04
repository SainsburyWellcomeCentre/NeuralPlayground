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
        super().__init__(environment_name, **env_kwargs)
        self.number_object= env_kwargs['number_object']
        self.room_width = env_kwargs['room_width']
        self.room_depth = env_kwargs['room_depth']
        self.state_density = env_kwargs['state_density']
        self.reset()

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

    def reset(self):
        self.global_steps = 0
        self.global_time = 0
        self.objects = np.zeros(shape=(self.n_states, self.number_object))
        self.generate_objects()

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
        state = self.pos_to_state(pos)
        object = self.objects[state]
        return state, object

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
