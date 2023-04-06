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
        self.room_widths = mod_kwargs['room_widths']
        self.room_depths = mod_kwargs['room_depths']
        self.state_densities = mod_kwargs['state_densities']
        self.pars = copy.deepcopy(params)
        self.tem = model.Model(self.pars)
        self.batch_size = mod_kwargs['batch_size']
        self.n_envs_save = 4
        self.n_states = [int(self.room_widths[i] * self.room_depths[i] * self.state_densities[i]) for i in range(self.batch_size)]
        self.actions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        self.n_actions = len(self.actions)
        self.final_model_input = None

        self.prev_observations = None
        self.reset()

    def reset(self):
        """
        initialise model and associated variables for training, set n_walk=-1 initially to account for the lack of
        actions at initialisation
        """
        self.tem = model.Model(self.pars)
        self.initialise()
        self.n_walk = -1
        self.final_model_input = None
        self.obs_history = []
        self.walk_actions = []
        self.prev_action = None
        self.prev_observation = None
        self.prev_actions = [[None,None] for _ in range(self.batch_size)]
        self.prev_observations = [[-1,-1,[float('inf'), float('inf')]] for _ in range(self.batch_size)]

    def act(self, observation, policy_func=None):
        new_action = self.action_policy()
        if observation[0] == self.prev_observation[0]:
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
        locations = [env[0] for env in observations]
        all_allowed = True
        new_actions = []
        for i, loc in enumerate(locations):
            if loc == self.prev_observations[i][0]:
                all_allowed = False
                break

        if all_allowed:
            self.walk_actions.append(self.prev_actions.copy())
            self.obs_history.append(self.prev_observations.copy())
            for batch in range(self.pars['batch_size']):
                new_actions.append(self.action_policy())
            self.prev_actions = new_actions
            self.prev_observations = observations
            self.n_walk += 1

        elif not all_allowed:
            for i, loc in enumerate(locations):
                if loc == self.prev_observations[i][0]:
                    new_actions.append(self.action_policy())
                else:
                    new_actions.append(self.prev_actions[i])
            self.prev_actions = new_actions

        return new_actions

    def update(self):
        """
        Compute forward pass through model, updating weights, calculating TEM variables and collecting losses / accuracies
        """
        self.iter = int((len(self.obs_history) / 20) - 1)
        # print(self.iter)
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
        self.eta_new, self.lambda_new, self.p2g_scale_offset, self.lr, self.walk_length_center, loss_weights = parameters.parameter_iteration(self.iter, self.pars)
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
        self.final_model_input = model_input

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
        if self.iter % 10 == 0:
            # Write series of messages to logger from this backprop iteration
            self.logger.info('Finished backprop iter {:d} in {:.2f} seconds.'.format(self.iter, time.time() - start_time))
            self.logger.info('Loss: {:.2f}. <p_g> {:.2f} <p_x> {:.2f} <x_gen> {:.2f} <x_g> {:.2f} <x_p> {:.2f} <g> {:.2f} <reg_g> {:.2f} <reg_p> {:.2f}'.format(
                    loss.detach().numpy(), *plot_loss))
            self.logger.info('Accuracy: <p> {:.2f}% <g> {:.2f}% <gt> {:.2f}%'.format(acc_p, acc_g, acc_gt))
            self.logger.info('Parameters: <max_hebb> {:.2f} <eta> {:.2f} <lambda> {:.2f} <p2g_scale_offset> {:.2f}'.format(
                np.max(np.abs(self.prev_iter[0].M[0].numpy())), self.tem.hyper['eta'], self.tem.hyper['lambda'], self.tem.hyper['p2g_scale_offset']))
            self.logger.info('Weights:' + str([w for w in loss_weights.numpy()]))
            self.logger.info(' ')

    def initialise(self):
        # Create directories for storing all information about the current run
        self.run_path, self.train_path, self.model_path, self.save_path, self.script_path, self.envs_path = utils.make_directories()
        # Save all python files in current directory to script directory
        curr_path = os.path.dirname(os.path.abspath(__file__))
        shutil.copy2(os.path.abspath(os.path.join(os.getcwd(), os.path.abspath(os.path.join(curr_path, os.pardir))))+'/agents/whittington_2020_extras/whittington_2020_model.py',
                     os.path.join(self.model_path, 'whittington_2020_model.py'))
        shutil.copy2(os.path.abspath(os.path.join(os.getcwd(), os.path.abspath(os.path.join(curr_path, os.pardir)))) + '/agents/whittington_2020_extras/whittington_2020_parameters.py',
                     os.path.join(self.model_path, 'whittington_2020_parameters.py'))
        # Save parameters
        np.save(os.path.join(self.save_path, 'params'), self.pars)
        # Create a tensor board to stay updated on training progress. Start tensorboard with tensorboard --logdir=runs
        self.writer = SummaryWriter(self.train_path)
        # Create a logger to write log output to file
        self.logger = utils.make_logger(self.run_path)
        # Make an ADAM optimizer for TEM
        self.adam = torch.optim.Adam(self.tem.parameters(), lr=self.pars['lr_max'])
        # Initialise whether a state has been visited for each world
        self.visited = [[False for _ in range(self.n_states[env])] for env in
                        range(self.pars['batch_size'])]
        # Initialise the previous iteration as None: we start from the beginning of the walk, so there is no previous iteration yet
        self.prev_iter = None

    def action_policy(self):
        """
        Random action policy that selects an action to take from [up, down, left, right]
        """
        arrow = self.actions
        index = np.random.choice(len(arrow))
        action = arrow[index]
        return action

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
        poss_values = self.actions
        for steps in actions:
            env_list = []
            for action in steps:
                env_list.append(poss_values.index(list(action)))
            action_values.append(env_list)
        return action_values

    def save_variables(self, forward):
        """
        Save all variables to file
        """
        # Save all variables to file
        np.save(self.save_path + '/n_states', self.n_states)
        np.save(self.save_path + '/n_actions', self.n_actions)
        np.save(self.save_path + '/env_dims', (self.room_widths, self.room_depths))
        torch.save(self.environments, self.save_path + '/environments')

        g, p = self.rate_maps(forward)
        torch.save(g, self.save_path + '/g_all')
        torch.save(p, self.save_path + '/p_all')

        correct_model, correct_node, correct_edge = self.compare_to_agents(forward)
        torch.save((correct_model, correct_node, correct_edge), self.save_path + '/correct_all')

        zero_shot = self.zero_shot(forward)
        torch.save(zero_shot, self.save_path + '/zero_shot')

    def collect_environment_info(self, model_input):
        self.final_model_input = []
        self.environments = [[], self.n_actions, self.n_states[0], len(self.obs_history[-1][0][1])]

        single_index = [[model_input[step][0][0]] for step in range(len(model_input))]
        single_obs = [torch.unsqueeze(model_input[step][1][0], dim=0) for step in range(len(model_input))]
        single_action = [[model_input[step][2][0]] for step in range(len(model_input))]
        single_model_input = [[single_index[step], single_obs[step], single_action[step]]for step in range(len(model_input))]
        self.final_model_input.extend(single_model_input)
        for step in range(len(model_input)):
            id = single_model_input[step][0][0]['id']
            if not any(d['id'] == id for d in self.environments[0]):
                loc_dict = {'id': id, 'observation': np.argmax(single_model_input[step][1]),
                            'x': history[step][0][-1][0], 'y': history[step][0][-1][1], 'shiny': None}
                self.environments[0].append(loc_dict)

        return self.environments

    def rate_maps(self, forward):
        # Store location x cell firing rate matrix for abstract and grounded location representation across environments
        all_g, all_p = [], []
        # Go through environments and collect firing rates in each
        for env_i in range(self.tem.hyper['batch_size']):
            # Collect grounded location/hippocampal/place cell representation during walk: separate into frequency modules, then locations
            p = [[[] for loc in range(self.n_states[env_i])] for f in range(self.pars['n_f'])]
            # Collect abstract location/entorhinal/grid cell representation during walk: separate into frequency modules, then locations
            g = [[[] for loc in range(self.n_states[env_i])] for f in range(self.pars['n_f'])]
            # In each step, concatenate the representations to the appropriate list
            for step in forward:
                # Run through frequency modules and append the firing rates to the correct location list
                for f in range(self.pars['n_f']):
                    g[f][step.g[env_i]['id']].append(step.g_inf[f][env_i].detach().numpy())
                    p[f][step.g[env_i]['id']].append(step.p_inf[f][env_i].detach().numpy())

            for cells, n_cells in zip([p, g], [self.pars['n_p'], self.pars['n_g']]):
                for f, frequency in enumerate(cells):
                    # Average across visits of the each location, but only the second half of the visits so model roughly know the environment
                    for l, location in enumerate(frequency):
                        frequency[l] = sum(location[int(len(location) / 2):]) / len(
                            location[int(len(location) / 2):]) if len(
                            location[int(len(location) / 2):]) > 0 else np.zeros(n_cells[f])
                    # Then concatenate the locations to get a [locations x cells for this frequency] matrix
                    cells[f] = np.stack(frequency, axis=0)
            # Append the final average representations of this environment to the list of representations across environments
            all_g.append(g)
            all_p.append(p)
        return all_g, all_p

    def compare_to_agents(self, forward, include_stay_still=False):
        # Store for each environment for each step whether is was predicted correctly by the model, and by a perfect node and perfect edge agent
        all_correct_model, all_correct_node, all_correct_edge = [], [], []
        # Run through environments and check for correct or incorrect prediction
        for env_i in range(self.tem.hyper['batch_size']):
            # Keep track for each location whether it has been visited
            location_visited = np.full(self.n_states[env_i], False)
            # And for each action in each location whether it has been taken
            action_taken = np.full((self.n_states[env_i], self.n_actions), False)
            # Make array to list whether the observation was predicted correctly or not for the model
            correct_model = []
            # And the same for a node agent, that picks a random observation on first encounter of a node, and the correct one every next time
            correct_node = []
            # And the same for an edge agent, that picks a random observation on first encounter of an edge, and the correct one every next time
            correct_edge = []
            # Get the very first iteration
            prev_iter = forward[0]
            # Run through iterations of forward pass to check when an action is taken for the first time
            for step in forward[1:]:
                # Get the previous action and previous location
                prev_a, prev_g = prev_iter.a[env_i], prev_iter.g[env_i]['id']
                # If the previous action was standing still: only count as valid transition standing still actions are included as zero-shot inference
                if self.pars['has_static_action'] and prev_a == 0 and not include_stay_still:
                    prev_a = None
                # Mark the location of the previous iteration as visited
                location_visited[prev_g] = True
                # Update model prediction for this step
                correct_model.append((torch.argmax(step.x_gen[2][env_i]) == torch.argmax(step.x[env_i])).numpy())
                # Update node agent prediction for this step: correct when this state was visited beofre, otherwise chance
                correct_node.append(True if location_visited[step.g[env_i]['id']] else np.random.randint(
                    self.pars['n_x']) == torch.argmax(step.x[env_i]).numpy())
                # Update edge agent prediction for this step: always correct if no action taken, correct when action leading to this state was taken before, otherwise chance
                correct_edge.append(
                    True if prev_a is None else True if action_taken[prev_g, prev_a] else np.random.randint(
                        self.pars['n_x']) == torch.argmax(step.x[env_i]).numpy())
                # Update the previous action as taken
                if prev_a is not None:
                    action_taken[prev_g, prev_a] = True
                # And update the previous iteration to the current iteration
                prev_iter = step
            # Add the performance of model, node agent, and edge agent for this environment to list across environments
            all_correct_model.append(correct_model)
            all_correct_node.append(correct_node)
            all_correct_edge.append(correct_edge)
        # Return list of prediction success for all three agents across environments
        return all_correct_model, all_correct_node, all_correct_edge

    def zero_shot(self, forward, include_stay_still=False):
        # Track for all opportunities for zero-shot inference if the predictions were correct across environments
        all_correct_zero_shot = []
        # Run through environments and check for zero-shot inference in each of them
        for env_i in range(self.tem.hyper['batch_size']):
            # Keep track for each location whether it has been visited
            location_visited = np.full(self.n_states[env_i], False)
            # And for each action in each location whether it has been taken
            action_taken = np.full((self.n_states[env_i], self.n_actions), False)
            # Get the very first iteration
            prev_iter = forward[0]
            # Make list that for all opportunities for zero-shot inference tracks if the predictions were correct
            correct_zero_shot = []
            # Run through iterations of forward pass to check when an action is taken for the first time
            for step in forward[1:]:
                # Get the previous action and previous location
                prev_a, prev_g = prev_iter.a[env_i], prev_iter.g[env_i]['id']
                # If the previous action was standing still: only count as valid transition standing still actions are included as zero-shot inference
                if self.pars['has_static_action'] and prev_a == 0 and not include_stay_still:
                    prev_a = None
                # Mark the location of the previous iteration as visited
                location_visited[prev_g] = True
                # Zero shot inference occurs when the current location was visited, but the previous action wasn't taken before
                if location_visited[step.g[env_i]['id']] and prev_a is not None and not action_taken[prev_g, prev_a]:
                    # Find whether the prediction was correct
                    correct_zero_shot.append(
                        (torch.argmax(step.x_gen[2][env_i]) == torch.argmax(step.x[env_i])).numpy())
                # Update the previous action as taken
                if prev_a is not None:
                    action_taken[prev_g, prev_a] = True
                # And update the previous iteration to the current iteration
                prev_iter = step
            # Having gone through the full forward pass for one environment, add the zero-shot performance to the list of all
            all_correct_zero_shot.append(correct_zero_shot)
        # Return lists of success of zero-shot inference for all environments
        return all_correct_zero_shot
