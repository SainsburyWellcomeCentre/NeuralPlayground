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
    def __init__(self, model_name: str = "TEM", **mod_kwargs):
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

        self.n_walk = 0
        self.obs_history = []
        self.walk_positions = []
        self.walk_actions = []

        self.initialise()

    def reset(self):
        self.tem = model.Model(self.pars)
        self.initialise()
        self.n_walk = 0
        self.obs_history = []
        self.walk_positions = []
        self.walk_actions = []

    def act(self, positions, policy_func=None):
        all_allowed = True
        actions = []
        for pos in positions:
            if None in pos:
                all_allowed = False
                break

        if all_allowed:
            for batch in range(self.pars['batch_size']):
                actions.append(self.action_policy())
            self.walk_actions.append(actions)
            self.obs_history.append(positions)
            self.walk_positions.append(positions)
            self.n_walk += 1

        elif not all_allowed:
            for i, pos in enumerate(positions):
                if None in pos:
                    actions.append(self.action_policy())
                else:
                    actions.append(self.walk_actions[-1][i])

        return actions

    def update(self):
        iter = len(self.obs_history)
        self.global_steps += 1
        positions = self.walk_positions
        actions = self.walk_actions
        self.walk_positions = []
        self.walk_actions = []
        self.n_walk = 0
        # Discretise (x,y) walk information
        locations = self.walk(positions)
        # Make observations
        observations = self.make_observations(locations)
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
        self.generate_objects()
        # Create directories for storing all information about the current run
        self.run_path, self.train_path, self.model_path, self.save_path, self.script_path, self.envs_path = utils.make_directories()
        # Save all python files in current directory to script directory
        shutil.copy2('/nfs/nhome/live/lhollingsworth/Documents/NeuralPlayground/NPG/EHC_model_comparison/neuralplayground/agents/TEM_extras/TEM_model.py',
                     os.path.join(self.model_path, 'TEM_model.py'))
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
        arrow = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        index = np.random.choice(len(arrow))
        action = arrow[index]
        return action

    def generate_objects(self):
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
        locations = []
        for step in positions:
            env_locations = []
            for env_step in step:
                index = self.discretise(env_step)
                env_locations.append({'id':index, 'shiny':None})
            locations.append(env_locations)
        return list(locations)

    def discretise(self, step):
        diff = self.xy_combination - step[np.newaxis, ...]
        dist = np.sum(diff ** 2, axis=2).T
        index = np.argmin(dist)
        return index

    def make_observations(self, locations):
        observations = []
        for i, step in enumerate(locations):
            env_observations = np.zeros(shape=(self.pars['batch_size'], self.pars['n_x']))
            for j in range(self.pars['batch_size']):
                env_observations[j] = self.objects[j][step[j]['id']]
            observations.append(env_observations)
        return observations

    def step_to_actions(self, actions):
        action_values = []
        # actions = np.reshape(actions, (pars['n_rollout'], pars['batch_size'], 2))
        poss_values = [[0,0],[0,-1],[0,1],[-1,0],[1,0]]
        for steps in actions:
            step_list = []
            for action in steps:
                step_list.append(poss_values.index(list(action)))
            action_values.append(step_list)
        return action_values