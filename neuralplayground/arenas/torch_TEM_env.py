import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random
from .simple2d import Simple2D
from ..agents.TEM_extras.TEM_parameters import parameters
from ..agents.TEM_extras.TEM_utils import *

pars_orig = parameters()
pars = pars_orig.copy()


class TEM_env(Simple2D):
    def __init__(self, environment_name='TEM_env', **env_kwargs):
        super().__init__(environment_name, **env_kwargs)
        self.room_width = abs(env_kwargs['arena_x_limits'][0] - env_kwargs['arena_x_limits'][1])
        self.room_depth = abs(env_kwargs['arena_y_limits'][0] - env_kwargs['arena_y_limits'][1])
        self.state_density = env_kwargs['state_density']

        # Variables for discretised (SR) state space
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

    def batch_reset(self, normalize_step=False, random_state=False, custom_state=None):
        self.global_steps = 0
        self.history = [[] for _ in range(pars['batch_size'])]
        self.states = np.zeros(shape=(pars['batch_size'], 2))
        locations = []
        for env in range(pars['batch_size']):
            if random_state:
                self.states[env] = [np.random.uniform(low=self.arena_limits[0, 0], high=self.arena_limits[0, 1]),
                              np.random.uniform(low=self.arena_limits[1, 0], high=self.arena_limits[1, 1])]
            else:
                self.states[env] = [0, 0]
            self.states[env] = np.array(self.states[env])

            if custom_state is not None:
                self.states[env] = custom_state

            locations.append(self.discretise(self.states[env]))
        return self.states, locations

    def batch_step(self, actions, normalize_step=False):
        observations = []
        states = []
        batch_history = []
        all_allowed = True
        for batch in range(pars['batch_size']):
            if normalize_step:
                action = actions[batch]
                new_state = self.states[batch] + [self.agent_step_size * direction for direction in action]
            else:
                action = actions[batch]
                new_state = self.states[batch] + action
            new_state, valid_action = self.validate_action(self.states[batch], action, new_state)
            reward = self.reward_function(action, self.states[batch])
            transition = {"action": action, "state": self.states[batch].copy(), "next_state": new_state,
                          "reward": reward, "step": self.global_steps}
            batch_history.append(transition)
            if valid_action:
                all_allowed = False
                observations.append([None, None])
                states.append(None)
            else:
                observations.append(new_state)
                states.append(self.discretise(new_state))

        if all_allowed:
            self.states = observations
            (self.history[i].extend(batch_history[i]) for i in range(pars['batch_size']))

        return observations, states

    def generate_test_environment(self):
        shiny = None if pars['shiny_rate'] ==0 else 0
        self.n_states = (self.room_width*self.room_depth)*self.state_density
        n_x = self.room_width*self.state_density
        n_y = self.room_depth*self.state_density

        all_pos = np.zeros(shape=(n_x, n_y, 2))
        for i in range(n_x):
            for j in range(n_y):
                all_pos[i,j] = [self.arena_x_limits[0]+j+0.5, self.arena_y_limits[1]-i-0.5]

        locations = self.generate_locations(np.reshape(all_pos, (n_x*n_y, 2)))

        return locations, 5, self.n_states, pars['n_x'], shiny

    def generate_test_objects(self):
        poss_objects = np.zeros(shape=(pars['n_x'], pars['n_x']))
        for i in range(pars['n_x']):
            for j in range(pars['n_x']):
                if j == i:
                    poss_objects[i][j] = 1
        objects = []
        objects = np.zeros(shape=(self.n_states, pars['n_x']))
        objects_index = np.zeros(shape=(self.n_states))
        # Generate landscape of objects in each environment
        for i in range(self.n_states):
            rand = random.randint(0, pars['n_x'] - 1)
            objects[i, :] = poss_objects[rand]
            objects_index[i] = rand
        return objects, objects_index

    def generate_locations(self, positions):
        locations = []
        obs, obs_index = self.generate_test_objects()
        self.resolution_w = int(self.state_density * self.room_width)
        self.resolution_d = int(self.state_density * self.room_depth)
        self.x_array = np.linspace(-self.room_width / 2 + 0.5, self.room_width / 2 - 0.5, num=self.resolution_w)
        self.y_array = np.linspace(self.room_depth / 2 - 0.5, -self.room_depth / 2 + 0.5, num=self.resolution_d)
        self.mesh = np.array(np.meshgrid(self.x_array, self.y_array))
        self.xy_combination = np.array(np.meshgrid(self.x_array, self.y_array)).T
        self.ws = int(self.room_width * self.state_density)
        self.hs = int(self.room_depth * self.state_density)
        node_layout = np.arange(self.n_states).reshape(self.room_width, self.room_depth)

        poss_actions = [[0,0], [0,1], [0,-1], [1,0], [-1,0]]

        for i, step in enumerate(positions):
            location = {'id':None, 'observation':None, 'x':None, 'y':None, 'in_locations':None, 'in_degree':None,
                        'out_locations':None, 'out_degree':None, 'actions':None, 'shiny':None}
            index = self.discretise(step)

            valid_actions = []
            valid_next_locs = []

            probs = []
            transitions = []
            for j, action in enumerate(poss_actions):
                transition = np.zeros(shape=(self.ws * self.hs))
                new_state = np.array(step) + np.array(action)
                new_state, valid_action = self.check_action(step, action, new_state)
                if valid_action == False:
                    valid_next_loc = self.discretise(new_state)
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
            location['shiny'] = True if pars['shiny_rate'] > 0 else None
            locations.append(location)
        return locations

    def discretise(self, step):
        diff = self.xy_combination - step[np.newaxis, ...]
        dist = np.sum(diff ** 2, axis=2).T
        index = np.argmin(dist)
        return index

    def check_action(self, pre_state, action, new_state):
        """

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
        valid_action: bool
            True if the change in state cross a wall
        """
        valid_action = False
        for wall in self.wall_list:
            new_state, new_valid_action = check_wall(pre_state=pre_state, new_state=new_state, wall=wall)
            valid_action = new_valid_action or valid_action
        return new_state, valid_action

    def generate_walks(self, env, walk_length=10, n_walk=100, repeat_bias_factor=2):
        # Generate walk by sampling actions accoring to policy, then next state according to graph
        walks = [] # This is going to contain a list of (state, observation, action) tuples
        for currWalk in range(n_walk):
            new_walk = []
            # If shiny hasn't been specified: there are no shiny objects, generate default policy
            if env[4] is None:
                new_walk = self.walk_default(env, new_walk, walk_length, repeat_bias_factor)
            # If shiny was specified: use policy that uses shiny policy to approach shiny objects sequentially
            else:
                new_walk = self.walk_shiny(env, new_walk, walk_length, repeat_bias_factor)
            # Clean up walk a bit by only keep essential location dictionary entries
            for step in new_walk[:-1]:
                step[0] = {'id': step[0]['id'], 'shiny': step[0]['shiny']}
            # Append new walk to list of walks
            walks.append(new_walk)
        return walks

    def walk_default(self, env, walk, walk_length, repeat_bias_factor=2):
        # Finish the provided walk until it contains walk_length steps
        for curr_step in range(walk_length - len(walk)):
            # Get new location based on previous action and location
            new_location = self.get_location(env, walk)
            # Get new observation at new location
            new_observation = self.get_observation(env, new_location)
            # Get new action based on policy at new location
            new_action = self.get_action(env, new_location, walk)
            # Append location, observation, and action to the walk
            walk.append([new_location, new_observation, new_action])
        # Return the final walk
        return walk

    def get_location(self, env, walk):
        # First step: start at random location
        if len(walk) == 0:
            new_location = np.random.randint(env[2])
        # Any other step: get new location from previous location and action
        else:
            new_location = int(np.flatnonzero(np.cumsum(walk[-1][0]['actions'][walk[-1][2]]['transition'])>np.random.rand())[0])
        # Return the location dictionary of the new location
        return env[0][new_location]

    def get_observation(self, env, new_location):
        # Find sensory observation for new state, and store it as one-hot vector
        new_observation = np.eye(env[3])[new_location['observation']]
        # Create a new observation by converting the new observation to a torch tensor
        new_observation = torch.tensor(new_observation, dtype=torch.float).view((new_observation.shape[0]))
        # Return the new observation
        return new_observation

    def get_action(self, env, new_location, walk, repeat_bias_factor=2):
        # Build policy from action probability of each action of provided location dictionary
        policy = np.array([action['probability'] for action in new_location['actions']])
        # Add a bias for repeating previous action to walk in straight lines, only if (this is not the first step) and (the previous action was a move)
        policy[[] if len(walk) == 0 or new_location['id'] == walk[-1][0]['id'] else walk[-1][2]] *= repeat_bias_factor
        # And renormalise policy (note that for unavailable actions, the policy was 0 and remains 0, so in that case no renormalisation needed)
        policy = policy / sum(policy) if sum(policy) > 0 else policy
        # Select action in new state
        new_action = int(np.flatnonzero(np.cumsum(policy)>np.random.rand())[0])
        # Return the new action
        return new_action

    def plot_batch_trajectory(self, batch=None, history_data=None, ax=None, return_figure=False):
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

        for batch in range(pars['batch_size']):
            if len(history_data[batch]) != 0:
                state_history = [s["state"] for s in history_data[batch]]
                next_state_history = [s["next_state"] for s in history_data[batch]]
                starting_point = state_history[0]
                ending_point = next_state_history[-1]

                cmap = mpl.cm.get_cmap("plasma")
                norm = plt.Normalize(0, len(state_history))

                aux_x = []
                aux_y = []
                for i, s in enumerate(state_history):
                    x_ = [s[0], next_state_history[i][0]]
                    y_ = [s[1], next_state_history[i][1]]
                    aux_x.append(s[0])
                    aux_y.append(s[1])
                    ax.plot(x_, y_, "-", color=cmap(norm(i)), alpha=0.6)

                # ax.set_xticks([])
                # ax.set_yticks([])
                sc = ax.scatter(aux_x, aux_y, c=np.arange(len(aux_x)), vmin=0, vmax=len(aux_x), cmap="plasma", alpha=0.6, s=0.1)
        cbar = plt.colorbar(sc, ax=ax,ticks = [0, len(state_history)])
        cbar.ax.set_ylabel('N steps', rotation=270,fontsize=16)
        cbar.ax.set_yticklabels([0,len(state_history)],fontsize=16)
        if return_figure:
            return f, ax
        else:
            return ax
