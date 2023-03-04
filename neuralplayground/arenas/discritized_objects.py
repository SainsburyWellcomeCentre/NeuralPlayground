import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random

from .simple2d import Simple2D

class DiscreteObjectEnvironment(Simple2D):

    def __init__(self, environment_name='DiscreteObject', **env_kwargs):
        super().__init__(environment_name, **env_kwargs)
        self.number_object= env_kwargs['number_object']
        self.room_width = env_kwargs['room_width']
        self.room_depth = env_kwargs['room_depth']
        self.state_density = env_kwargs['state_density']

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
        self.node_layout = np.arange(self.n_states).reshape(self.room_width, self.room_depth)

    def reset(self, normalize_step=False, random_state=False, custom_state=None):
        """
        Parameters:
        ------
            normalize_step: boolean
                Whether the steps taken are of unit length or scaled by the agent step size
            random_state: boolean
                Wether the start of each trajectory is random or whether they all originate at [0,0]
            custom_state: boolean
                Whether a custom start location is given for the trajectories.
        Returns:
        ------
        """
        self.global_steps = 0
        locations = []
        if random_state:
            self.states[env] = [np.random.uniform(low=self.arena_limits[0, 0], high=self.arena_limits[0, 1]),
                              np.random.uniform(low=self.arena_limits[1, 0], high=self.arena_limits[1, 1])]
        else:
            self.states[env] = [0, 0]
            self.states[env] = np.array(self.states[env])
        if custom_state is not None:
            self.states[env] = custom_state
        locations.append(self.obs_to_state(self.states[env]))
        return self.states, locations

    def generate_environment(self):
        self.n_states = (self.room_width*self.room_depth)*self.state_density
        all_pos = np.zeros(shape=( self.resolution_d, self.resolution_w, 2))
        for i in range( self.resolution_w):
            for j in range(self.resolution_d):
                all_pos[i,j] = [self.arena_x_limits[0]+j+0.5, self.arena_y_limits[1]-i-0.5]
        locations = self.generate_locations(np.reshape(all_pos, ( self.resolution_w* self.resolution_d, 2)))
        return locations, self.n_states,

    def generate_objects(self):
        poss_objects = np.zeros(shape=(self.number_object,self.number_object))
        for i in range(self.number_object):
            for j in range(self.number_object):
                if j == i:
                    poss_objects[i][j] = 1
        objects = []
        objects = np.zeros(shape=(self.n_states, self.number_object))
        objects_index = np.zeros(shape=(self.n_states))
        # Generate landscape of objects in each environment
        for i in range(self.n_states):
            rand = random.randint(0, self.number_object - 1)
            objects[i, :] = poss_objects[rand]
            objects_index[i] = rand
        return objects, objects_index

    def generate_locations(self, positions):
        locations = []
        obs, obs_index = self.generate_objects()

        poss_actions = [[0,0], [0,1], [0,-1], [1,0], [-1,0]]

        for i, step in enumerate(positions):
            location = {'id':None, 'observation':None, 'x':None, 'y':None, 'in_locations':None, 'in_degree':None,
                        'out_locations':None, 'out_degree':None, 'actions':None, 'shiny':None}
            index = self.pos_to_state(step)
            valid_next_locs = []

            probs = []
            transitions = []
            for j, action in enumerate(poss_actions):
                transition = np.zeros(shape=(self.ws * self.hs))
                new_state = np.array(step) + np.array(action)
                new_state, valid_action = self.check_action(step, action, new_state)
                if valid_action == False:
                    valid_next_loc = self.obs_to_state(new_state)
                    valid_next_locs.append(valid_next_loc)
                    transition[valid_next_loc] = 1
                    probs.append(1)
                else:
                    probs.append(0)
                transitions.append(transition.tolist())
            probs = [prob / len(valid_next_locs) for prob in probs]
            action_dict = [{'id': k, 'transition': transitions[k], 'probability': probs[k]} for k in  range(len(poss_actions))]

            location['id'] = index
            location['observation'] = int(obs_index[index])
            location['x'], location['y'] = step[0], step[1]
            location['in_locations'] = location['out_locations'] = valid_next_locs
            location['in_degree'] = location['out_degree'] = len(valid_next_locs)
            location['actions'] = action_dict
            location['shiny'] = False
            locations.append(location)
        return locations

    def pos_to_state(self, step):
        diff = self.xy_combination - step[np.newaxis, ...]
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
