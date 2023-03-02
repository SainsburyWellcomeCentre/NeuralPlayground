import copy
import sys
sys.path.append("../")

import numpy as np
import torch
import time
import os
import shutil
from torch.utils.tensorboard import SummaryWriter

# Custom modules
import neuralplayground.agents.TEM_extras.TEM_utils as utils
import neuralplayground.agents.TEM_extras.TEM_parameters as parameters
from .agent_core import AgentCore
import neuralplayground.agents.TEM_extras.TEM_model as model

class TEM(AgentCore):
    def __init__(self, model_name: str = "TEM", params=None):
        super().__init__()
        self.pars = copy.deepcopy(params)
        self.tem = model.Model(self.pars)
        self.initialise()

    def act(self, iter, locations, observations, actions):
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
        model_input = [[locations[i].tolist(), torch.from_numpy(np.reshape(observations, (20, 16, 45))[i]).type(torch.float32),
                 np.reshape(actions, (20, 16))[i].tolist()] for i in range(self.pars['n_rollout'])]

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