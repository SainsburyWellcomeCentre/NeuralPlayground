import copy
import sys
sys.path.append("../")

import numpy as np
import torch
import random
import time
import os
import shutil
from torch.utils.tensorboard import SummaryWriter

# Custom modules
import neuralplayground.agents.whittington_2020_extras.whittington_2020_utils as utils
import neuralplayground.agents.whittington_2020_extras.whittington_2020_parameters as parameters
from .agent_core import AgentCore
import neuralplayground.agents.whittington_2020_extras.whittington_2020_model as model

class Whittington2020(AgentCore):
    """
        Implementation of TEM 2020 by James C.R. Whittington, Timothy H. Muller, Shirley Mark, Guifen Chen, Caswell Barry,
        Neil Burgess, Timothy E.J. Behrens. The Tolman-Eichenbaum Machine: Unifying Space and Relational Memory through
        Generalization in the Hippocampal Formation https://doi.org/10.1016/j.cell.2020.10.024.
        ----
        Attributes
        ---------
        mod_kwargs : dict
            Model parameters
            params: dict
                contains the majority of parameters used by the model and environment
            room_width: float
                room width specified by the environment (see examples/examples/whittington_2020_example.ipynb)
            room_depth: float
                room depth specified by the environment (see examples/examples/whittington_2020_example.ipynb)
            state_density: float
                density of agent states (should be proportional to the step-size)
            tem: class
                TEM model

        Methods
        ---------
        reset(self):
            initialise model and associated variables for training
        def initialise(self):
            generate random distribution of objects and intialise optimiser, logger and relevant variables
        act(self, positions, policy_func):
            generates batch of random actions to be passed into the environment. If the returned positions are allowed,
            they are saved along with corresponding actions
        update(self):
            Perform forward pass of model and calculate losses and accuracies
        action_policy(self):
            random action policy that picks from [up, down, left right]
        discretise(self, step):
            convert (x,y) position into discrete location
        walk(self, positions):
            convert continuous positions into sequence of discrete locations
        make_observations(self, locations):
            observe what randomly distributed object is located at each position of a walk
        step_to_actions(self, actions):
            convert (x,y) action information into an integer value
        """
    def __init__(self, model_name: str = "TEM", **mod_kwargs):
        """
        Parameters
        ----------
        model_name : str
           Name of the specific instantiation of the ExcInhPlasticity class
        mod_kwargs : dict
            params: dict
                contains the majority of parameters used by the model and environment
            room_width: float
                room width specified by the environment (see examples/examples/whittington_2020_example.ipynb)
            room_depth: float
                room depth specified by the environment (see examples/examples/whittington_2020_example.ipynb)
            state_density: float
                density of agent states (should be proportional to the step-size)
        """
        super().__init__()
        params = mod_kwargs['params']
        self.room_width = abs(mod_kwargs['room_width'][0] - mod_kwargs['room_width'][1])
        self.room_depth = abs(mod_kwargs['room_depth'][0] - mod_kwargs['room_depth'][1])
        self.state_density = mod_kwargs['state_density']
        self.pars = copy.deepcopy(params)
        self.tem = model.Model(self.pars)

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

        self.reset()

    def reset(self):
        """
        initialise model and associated variables for training, set n_walk=-1 initially to account for the lack of
        actions at initialisation
        """
        self.tem = model.Model(self.pars)
        self.initialise()
        self.n_walk = -1
        self.obs_history = []
        self.walk_actions = []
        self.prev_actions = []
        self.prev_observations = []

        self.prev_action = None
        self.prev_observation = None

    def act(self, observation, policy_func=None):
        new_action = self.action_policy()
        if None in observation[-1]:
            self.prev_action = new_action
        else:
            self.walk_actions.append(self.prev_action)
            self.obs_history.append(self.prev_observation)
            self.prev_action = new_action
            self.prev_observation = observation
            self.n_walk += 1

        return new_action

    def batch_act(self, observations, policy_func=None):
        """
        The base model executes one of four action (up-down-right-left) with equal probability.
        This is used to move on the rectangular environment states space (transmat).
        This is done for a batch of 16 environments.
        Parameters
        ----------
        positions: array (16,2)
            Observation from the environment class needed to choose the right action (Here the position).
        Returns
        -------
        action : array (16,2)
            Action values (Direction of the agent step) in this case executes one of four action (up-down-right-left) with equal probability.
        """
        all_allowed = True
        new_actions = []
        positions = [env[2] for env in observations]
        for pos in positions:
            if None in pos:
                all_allowed = False
                break

        if all_allowed:
            self.walk_actions.append(self.prev_actions)
            self.obs_history.append(self.prev_observations)
            for batch in range(self.pars['batch_size']):
                new_actions.append(self.action_policy())
            self.prev_actions = new_actions
            self.prev_observations = observations
            self.n_walk += 1

        elif not all_allowed:
            for i, pos in enumerate(positions):
                if None in pos:
                    new_actions.append(self.action_policy())
                else:
                    new_actions.append(self.prev_actions[i])
            self.prev_actions = new_actions

        return new_actions

    def update(self):
        """
        Compute forward pass through model, updating weights, calculating TEM variables and collecting losses / accuracies
        """
        iter = int((len(self.obs_history) / 20) - 1)
        print(iter)
        self.global_steps += 1
        history = self.obs_history[-self.pars['n_rollout']:]
        locations = [[{'id': env_step[0], 'shiny': None} for env_step in step] for step in history]
        observations = [[env_step[1] for env_step in step] for step in history]
        actions = self.walk_actions[-self.pars['n_rollout']:]
        self.walk_positions = []
        self.walk_actions = []
        self.n_walk = 0
        # Convert action vectors to action values
        action_values = self.step_to_actions(actions)
        # Get start time for function timing
        start_time = time.time()
        # Get updated parameters for this backprop iteration
        self.eta_new, self.lambda_new, self.p2g_scale_offset, self.lr, self.walk_length_center, loss_weights = parameters.parameter_iteration(iter, self.pars)
        # Update eta and lambda
        self.tem.hyper['eta'] = self.eta_new
        self.tem.hyper['lambda'] = self.lambda_new
        # Update scaling of offset for variance of inferred grounded position
        self.tem.hyper['p2g_scale_offset'] = self.p2g_scale_offset
        # Update learning rate (the neater torch-way of doing this would be a scheduler, but this is quick and easy)
        for param_group in self.adam.param_groups:
            param_group['lr'] = self.lr

        # Collect all information in walk variable
        model_input = [[locations[i], torch.from_numpy(np.reshape(observations, (20, 16, 45))[i]).type(torch.float32),
                 np.reshape(action_values, (20, 16))[i].tolist()] for i in range(self.pars['n_rollout'])]

        forward = self.tem(model_input, self.prev_iter)

        # Accumulate loss from forward pass
        loss = torch.tensor(0.0)
        # Make vector for plotting losses
        plot_loss = 0
        # Collect all losses / variables
        for ind, step in enumerate(forward):
            # Make list of losses included in this step
            step_loss = []
            # Only include loss for locations that have been visited before
            for env_i, env_visited in enumerate(self.visited):
                if env_visited[step.g[env_i]['id']]:
                    step_loss.append(loss_weights * torch.stack([l[env_i] for l in step.L]))
                else:
                    env_visited[step.g[env_i]['id']] = True
            # Stack losses in this step along first dimension, then average across that dimension to get mean loss for this step
            step_loss = torch.tensor(0) if not step_loss else torch.mean(torch.stack(step_loss, dim=0), dim=0)
            # Save all separate components of loss for monitoring
            plot_loss = plot_loss + step_loss.detach().numpy()
            # And sum all components, then add them to total loss of this step
            loss = loss + torch.sum(step_loss)

        # Reset gradients
        self.adam.zero_grad()
        # Do backward pass to calculate gradients with respect to total loss of this chunk
        loss.backward(retain_graph=True)
        # Then do optimiser step to update parameters of model
        self.adam.step()
        # Update the previous iteration for the next chunk with the final step of this chunk, removing all operation history
        self.prev_iter = [forward[-1].detach()]

        # Compute model accuracies
        acc_p, acc_g, acc_gt = np.mean([[np.mean(a) for a in step.correct()] for step in forward], axis=0)
        acc_p, acc_g, acc_gt = [a * 100 for a in (acc_p, acc_g, acc_gt)]
        # Log progress
        if iter % 10 == 0:
            # Write series of messages to logger from this backprop iteration
            self.logger.info('Finished backprop iter {:d} in {:.2f} seconds.'.format(iter, time.time() - start_time))
            self.logger.info('Loss: {:.2f}. <p_g> {:.2f} <p_x> {:.2f} <x_gen> {:.2f} <x_g> {:.2f} <x_p> {:.2f} <g> {:.2f} <reg_g> {:.2f} <reg_p> {:.2f}'.format(
                    loss.detach().numpy(), *plot_loss))
            self.logger.info('Accuracy: <p> {:.2f}% <g> {:.2f}% <gt> {:.2f}%'.format(acc_p, acc_g, acc_gt))
            self.logger.info('Parameters: <max_hebb> {:.2f} <eta> {:.2f} <lambda> {:.2f} <p2g_scale_offset> {:.2f}'.format(
                np.max(np.abs(self.prev_iter[0].M[0].numpy())), self.tem.hyper['eta'], self.tem.hyper['lambda'], self.tem.hyper['p2g_scale_offset']))
            self.logger.info('Weights:' + str([w for w in loss_weights.numpy()]))
            self.logger.info(' ')

        if iter % self.pars['save_interval'] == 0 and iter > 0:
            torch.save(self.tem.state_dict(), self.model_path + '/tem_' + str(iter) + '.pt')
            torch.save(self.tem.hyper, self.model_path + '/params_' + str(iter) + '.pt')

        if iter == self.pars['train_it'] - 1:
            # Save the final state of the model after training has finished
            torch.save(self.state_dict(), self.model_path + '/tem_' + str(iter) + '.pt')
            torch.save(self.tem.hyper, self.model_path + '/params_' + str(iter) + '.pt')

    def initialise(self):
        """
        Generate random distribution of objects for batch of environments. Initialise other
        """
        self.generate_objects()
        # Create directories for storing all information about the current run
        self.run_path, self.train_path, self.model_path, self.save_path, self.script_path, self.envs_path = utils.make_directories()
        # Save all python files in current directory to script directory
        curr_path = os.path.dirname(os.path.abspath(__file__))
        shutil.copy2(os.path.abspath(os.path.join(os.getcwd(), os.path.abspath(os.path.join(curr_path, os.pardir))))+'/agents/whittington_2020_extras/whittington_2020_model.py',
                     os.path.join(self.model_path, 'whittington_2020_model.py'))
        # Save parameters
        np.save(os.path.join(self.save_path, 'params'), self.pars)
        # Create a tensor board to stay updated on training progress. Start tensorboard with tensorboard --logdir=runs
        self.writer = SummaryWriter(self.train_path)
        # Create a logger to write log output to file
        self.logger = utils.make_logger(self.run_path)
        # Make an ADAM optimizer for TEM
        self.adam = torch.optim.Adam(self.tem.parameters(), lr=self.pars['lr_max'])
        # Initialise whether a state has been visited for each world
        self.visited = [[False for _ in range(self.pars['n_states_world'][env])] for env in
                        range(self.pars['batch_size'])]
        # Initialise the previous iteration as None: we start from the beginning of the walk, so there is no previous iteration yet
        self.prev_iter = None
        # Create space for full trajectory
        self.walks = []
        # Create position counts for agent coverage plots
        self.states = [self.pars['n_states_world'][self.pars['diff_env_batches_envs'][env]] for env in
                       range(self.pars['batch_size'])]

    def action_policy(self):
        """
        Random action policy that selects an action to take from [up, down, left, right]
        """
        arrow = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        index = np.random.choice(len(arrow))
        action = arrow[index]
        return action

    def generate_objects(self):
        """
        Generate a random distributions of n_x possible objects. Each is represented as a one-hot encoded vector and is
        what TEM sees at each location.
        """
        poss_objects = np.zeros(shape=(self.pars['n_x'], self.pars['n_x']))
        for i in range(self.pars['n_x']):
            for j in range(self.pars['n_x']):
                if j == i:
                    poss_objects[i][j] = 1
        self.objects = []
        for batch in range(self.pars['batch_size']):
            env_objects = np.zeros(shape=(self.n_states, self.pars['n_x']))
            # Generate landscape of objects in each environment
            for i in range(self.n_states):
                rand = random.randint(0, self.pars['n_x'] - 1)
                env_objects[i, :] = poss_objects[rand]
            self.objects.append(env_objects)

    def walk(self, positions):
        """
        Converts a trajectory in continuous 2D space to a sequence of discretised locations.
        ----
        Parameters:
        -------
            positions:
                list of (x,y) coordinates for each step in a walk (of length n_rollout)

        Returns:
        -------
            locations:
                list of location values (e.g. room of width and depth of 10 would be: [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                                                       10,11,12,13,14,15,16,17,18,19,
                                                                                                    ...
                                                                                       80,81,82,83,84,85,86,87,88,89,
                                                                                       90,91,92,93,94,95,96,97,98,99]
        """
        locations = []
        for step in positions:
            env_locations = []
            for env_step in step:
                index = self.discretise(env_step)
                env_locations.append({'id':index, 'shiny':None})
            locations.append(env_locations)
        return list(locations)

    def discretise(self, step):
        """
        Convert single (x,y) position into discrete location

        Parameters:
        -------
            step: (2,1)
                (x,y) position within continuous environment

        Returns:
        ------
            index: int
        """
        diff = self.xy_combination - step[np.newaxis, ...]
        dist = np.sum(diff ** 2, axis=2).T
        index = np.argmin(dist)
        return index

    def make_observations(self, locations):
        """
        Converts sequence of discrete locations to the one-hot objects present at those locations

        Parameters:
        ------
            locations: (16,1)
                sequence of discrete locations within an environment

        Returns:
        ------
            observations: (16,45)
                one-hot encoded objects present at each step of a trajectory
        """
        observations = []
        for i, step in enumerate(locations):
            env_observations = np.zeros(shape=(self.pars['batch_size'], self.pars['n_x']))
            for j in range(self.pars['batch_size']):
                env_observations[j] = self.objects[j][step[j]['id']]
            observations.append(env_observations)
        return observations

    def step_to_actions(self, actions):
        """
        Convert trajectory of (x,y) actions into integer values (i.e. from [[0,0],[-1,0],[1,0],[0,-1],[0,1]] to [0,1,2,3,4])

        Parameters:
        ------
            actions: (16,20,2)
                batch of 16 actions for each step in a walk of length 20

        Returns:
        ------
            action_values: (16,20,1)
                batch of 16 action values for each step in walk of length 20
        """
        action_values = []
        # actions = np.reshape(actions, (pars['n_rollout'], pars['batch_size'], 2))
        poss_values = [[0,0],[0,-1],[0,1],[-1,0],[1,0]]
        for steps in actions:
            env_list = []
            for action in steps:
                env_list.append(poss_values.index(list(action)))
            action_values.append(env_list)
        return action_values