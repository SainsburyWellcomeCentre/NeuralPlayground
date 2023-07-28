import copy
import os
import shutil
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import neuralplayground.agents.whittington_2020_extras.whittington_2020_model as model
import neuralplayground.agents.whittington_2020_extras.whittington_2020_parameters as parameters

# Custom modules
import neuralplayground.agents.whittington_2020_extras.whittington_2020_utils as utils
from neuralplayground.plotting.plot_utils import make_plot_rate_map

from .agent_core import AgentCore

sys.path.append("../")


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
        params = mod_kwargs["params"]
        self.room_widths = mod_kwargs["room_widths"]
        self.room_depths = mod_kwargs["room_depths"]
        self.state_densities = mod_kwargs["state_densities"]
        self.pars = copy.deepcopy(params)
        self.tem = model.Model(self.pars)
        self.batch_size = mod_kwargs["batch_size"]
        self.use_behavioural_data = mod_kwargs["use_behavioural_data"]
        self.n_envs_save = 4
        self.n_states = [
            int(self.room_widths[i] * self.room_depths[i] * self.state_densities[i]) for i in range(self.batch_size)
        ]
        self.poss_actions = [[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]]
        self.n_actions = len(self.poss_actions)
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
        self.walk_action_values = []
        self.prev_action = None
        self.prev_observation = None
        self.prev_actions = [[None, None] for _ in range(self.batch_size)]
        self.prev_observations = [[-1, -1, [float("inf"), float("inf")]] for _ in range(self.batch_size)]

    def act(self, observation, policy_func=None):
        """
        The base model executes one of four action (up-down-right-left) with equal probability.
        This is used to move on the rectangular environment states space (transmat).
        This is done for a single environment.
        Parameters
        ----------
        positions: array (16,2)
            Observation from the environment class needed to choose the right action (Here the position).
        Returns
        -------
        action : array (16,2)
            Action values (Direction of the agent step) in this case executes one of four action
            (up-down-right-left) with equal probability.
        """
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
        observations: array (16,3/4)
            Observation from the environment class needed to choose the right action
            (here the state ID and position). If behavioural data is used, the observation includes head direction.
        Returns
        -------
        new_actions : array (16,2)
            Action values (direction of the agent step) in this case executes one of four action
            (up-down-right-left) with equal probability.
        """

        if self.use_behavioural_data:
            state_diffs = [observations[i][0] - self.prev_observations[i][0] for i in range(self.batch_size)]
            new_actions = self.infer_action(state_diffs)
            self.walk_actions.append(new_actions)
            self.obs_history.append(self.prev_observations.copy())
            self.prev_observations = observations
            self.n_walk += 1

        elif not self.use_behavioural_data:
            locations = [env[0] for env in observations]
            all_allowed = True
            new_actions = []
            for i, loc in enumerate(locations):
                if loc == self.prev_observations[i][0] and self.prev_actions[i] != [0, 0]:
                    all_allowed = False
                    break

            if all_allowed:
                self.walk_actions.append(self.prev_actions.copy())
                self.obs_history.append(self.prev_observations.copy())
                for batch in range(self.pars["batch_size"]):
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
        Compute forward pass through model, updating weights, calculating TEM variables and collecting
        losses / accuracies
        """
        self.iter = int((len(self.obs_history) / 20) - 1)
        self.global_steps += 1
        history = self.obs_history[-self.pars["n_rollout"] :]
        locations = [[{"id": env_step[0], "shiny": None} for env_step in step] for step in history]
        observations = [[env_step[1] for env_step in step] for step in history]
        actions = self.walk_actions[-self.pars["n_rollout"] :]
        self.n_walk = 0
        # Convert action vectors to action values
        action_values = self.step_to_actions(actions)
        self.walk_action_values.append(action_values)
        # Get start time for function timing
        start_time = time.time()
        # Get updated parameters for this backprop iteration
        (
            self.eta_new,
            self.lambda_new,
            self.p2g_scale_offset,
            self.lr,
            self.walk_length_center,
            loss_weights,
        ) = parameters.parameter_iteration(self.iter, self.pars)
        # Update eta and lambda
        self.tem.hyper["eta"] = self.eta_new
        self.tem.hyper["lambda"] = self.lambda_new
        # Update scaling of offset for variance of inferred grounded position
        self.tem.hyper["p2g_scale_offset"] = self.p2g_scale_offset
        # Update learning rate (the neater torch-way of doing this would be a scheduler, but this is quick and easy)
        for param_group in self.adam.param_groups:
            param_group["lr"] = self.lr

        # Collect all information in walk variable
        model_input = [
            [
                locations[i],
                torch.from_numpy(np.reshape(observations, (20, 16, 45))[i]).type(torch.float32),
                np.reshape(action_values, (20, 16))[i].tolist(),
            ]
            for i in range(self.pars["n_rollout"])
        ]
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
                if env_visited[step.g[env_i]["id"]]:
                    step_loss.append(loss_weights * torch.stack([i[env_i] for i in step.L]))
                else:
                    env_visited[step.g[env_i]["id"]] = True
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
            self.logger.info("Finished backprop iter {:d} in {:.2f} seconds.".format(self.iter, time.time() - start_time))
            self.logger.info(
                "Loss: {:.2f}. <p_g> {:.2f} <p_x> {:.2f} <x_gen> {:.2f} <x_g> {:.2f} <x_p> {:.2f} <g> {:.2f} \
                    <reg_g> {:.2f} <reg_p> {:.2f}".format(
                    loss.detach().numpy(), *plot_loss
                )
            )
            self.logger.info("Accuracy: <p> {:.2f}% <g> {:.2f}% <gt> {:.2f}%".format(acc_p, acc_g, acc_gt))
            self.logger.info(
                "Parameters: <max_hebb> {:.2f} <eta> {:.2f} <lambda> {:.2f} <p2g_scale_offset> {:.2f}".format(
                    np.max(np.abs(self.prev_iter[0].M[0].numpy())),
                    self.tem.hyper["eta"],
                    self.tem.hyper["lambda"],
                    self.tem.hyper["p2g_scale_offset"],
                )
            )
            self.logger.info("Weights:" + str([w for w in loss_weights.numpy()]))
            self.logger.info(" ")
        # Also store the internal state (all learnable parameters) and the hyperparameters periodically
        if self.iter % self.pars["save_interval"] == 0:
            torch.save(self.tem.state_dict(), self.model_path + "/tem_" + str(self.iter) + ".pt")
            torch.save(self.tem.hyper, self.model_path + "/params_" + str(self.iter) + ".pt")

        # Save the final state of the model after training has finished
        if self.iter == self.pars["train_it"] - 1:
            torch.save(self.tem.state_dict(), self.model_path + "/tem_" + str(self.iter) + ".pt")
            torch.save(self.tem.hyper, self.model_path + "/params_" + str(self.iter) + ".pt")

    def initialise(self):
        """
        Generate random distribution of objects and intialise optimiser, logger and relevant variables
        """
        # Create directories for storing all information about the current run
        (
            self.run_path,
            self.train_path,
            self.model_path,
            self.save_path,
            self.script_path,
            self.envs_path,
        ) = utils.make_directories()
        # Save all python files in current directory to script directory
        self.save_files()
        # Save parameters
        np.save(os.path.join(self.save_path, "params"), self.pars)
        # Create a tensor board to stay updated on training progress. Start tensorboard with tensorboard --logdir=runs
        self.writer = SummaryWriter(self.train_path)
        # Create a logger to write log output to file
        self.logger = utils.make_logger(self.run_path)
        # Make an ADAM optimizer for TEM
        self.adam = torch.optim.Adam(self.tem.parameters(), lr=self.pars["lr_max"])
        # Initialise whether a state has been visited for each world
        self.visited = [[False for _ in range(self.n_states[env])] for env in range(self.pars["batch_size"])]
        self.prev_iter = None

    def save_files(self):
        """
        Save all python files in current directory to script directory
        """
        curr_path = os.path.dirname(os.path.abspath(__file__))
        shutil.copy2(
            os.path.abspath(os.path.join(os.getcwd(), os.path.abspath(os.path.join(curr_path, os.pardir))))
            + "/agents/whittington_2020_extras/whittington_2020_model.py",
            os.path.join(self.script_path, "whittington_2020_model.py"),
        )
        shutil.copy2(
            os.path.abspath(os.path.join(os.getcwd(), os.path.abspath(os.path.join(curr_path, os.pardir))))
            + "/agents/whittington_2020_extras/whittington_2020_parameters.py",
            os.path.join(self.script_path, "whittington_2020_parameters.py"),
        )
        shutil.copy2(
            os.path.abspath(os.path.join(os.getcwd(), os.path.abspath(os.path.join(curr_path, os.pardir))))
            + "/agents/whittington_2020_extras/whittington_2020_analyse.py",
            os.path.join(self.script_path, "whittington_2020_analyse.py"),
        )
        shutil.copy2(
            os.path.abspath(os.path.join(os.getcwd(), os.path.abspath(os.path.join(curr_path, os.pardir))))
            + "/agents/whittington_2020_extras/whittington_2020_plot.py",
            os.path.join(self.script_path, "whittington_2020_plot.py"),
        )
        shutil.copy2(
            os.path.abspath(os.path.join(os.getcwd(), os.path.abspath(os.path.join(curr_path, os.pardir))))
            + "/agents/whittington_2020_extras/whittington_2020_utils.py",
            os.path.join(self.script_path, "whittington_2020_utils.py"),
        )
        return

    def action_policy(self):
        """
        Random action policy that selects an action to take from [stay, up, down, left, right]
        """
        arrow = self.poss_actions
        index = np.random.choice(len(arrow))
        action = arrow[index]
        return action

    def step_to_actions(self, actions):
        """
        Convert trajectory of (x,y) actions into integer values (i.e. from [[0,0],[0,-1],[1,0],[0,1],[-1,0]] to [0,1,2,3,4])

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
        for steps in actions:
            env_list = []
            for action in steps:
                env_list.append(self.poss_actions.index(list(action)))
            action_values.append(env_list)
        return action_values

    def infer_action(self, state_diffs):
        """
        Infers the action taken between state indices based on the difference between states.

        Parameters
        ----------
        state_diff: int
            The difference between the state indices.
        environment_width: int
            The width of the environment (number of states per row).

        Returns
        -------
        action: str
            The inferred action ('N', 'S', 'W', or 'E') based on the state difference.
        """
        actions = []
        for i in range(self.batch_size):
            if state_diffs[i] == -self.room_widths[i]:
                actions.append([0, 1])
            elif state_diffs[i] == self.room_widths[i]:
                actions.append([0, -1])
            elif state_diffs[i] == -1:
                actions.append([-1, 0])
            elif state_diffs == 1:
                actions.append([1, 0])
            else:
                actions.append([0, 0])

        return actions

    def collect_final_trajectory(self):
        """
        Collect the final trajectory of the agent, including the locations, observations and actions taken.
        """
        final_model_input = []
        environments = [[], self.n_actions, self.n_states[0], len(self.obs_history[-1][0][1])]
        history = self.obs_history[-self.n_walk :]
        locations = [[{"id": env_step[0], "shiny": None} for env_step in step] for step in history]
        observations = [[env_step[1] for env_step in step] for step in history]
        actions = self.walk_actions[-self.n_walk :]
        action_values = self.step_to_actions(actions)

        model_input = [
            [
                locations[i],
                torch.from_numpy(np.reshape(observations, (self.n_walk, 16, 45))[i]).type(torch.float32),
                np.reshape(action_values, (self.n_walk, 16))[i].tolist(),
            ]
            for i in range(self.n_walk)
        ]

        single_index = [[model_input[step][0][0]] for step in range(len(model_input))]
        single_obs = [torch.unsqueeze(model_input[step][1][0], dim=0) for step in range(len(model_input))]
        single_action = [[model_input[step][2][0]] for step in range(len(model_input))]
        single_model_input = [[single_index[step], single_obs[step], single_action[step]] for step in range(len(model_input))]
        final_model_input.extend(single_model_input)

        return final_model_input, history, environments

    def plot_rate_map(self, rate_maps):
        """
        Plot the TEM rate maps.

        Parameters
        ----------
        rate_maps: ndarray, shape (5, N)
            The rate maps for TEM, where N is the number of cells in each frequency.

        Returns
        -------
        figs: list
            A list of matplotlib figures containing the rate map plots for each frequency.
        axes: list
            A list of arrays of matplotlib axes containing the individual rate map plots for each frequency.
        """
        frequencies = ["Theta", "Delta", "Beta", "Gamma", "High Gamma"]
        figs = []
        axes = []

        for i in range(5):
            n_cells = rate_maps[0][i].shape[1]
            num_cols = 6  # Number of subplots per row
            num_rows = np.ceil(n_cells / num_cols).astype(int)

            # Create the figure for the current frequency
            fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 10))
            fig.suptitle(f"{frequencies[i]} Rate Maps", fontsize=16)

            # Create a single colorbar for the entire figure
            cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])

            # Create the subplots for the current frequency
            for j in range(n_cells):
                if j >= n_cells:
                    break
                ax_row = j // num_cols
                ax_col = j % num_cols

                # Get the rate map for the current cell and frequency
                rate_map = np.asarray(rate_maps[0][i]).T[j]

                # Reshape the rate map into a matrix
                rate_map_mat = np.reshape(rate_map, (self.room_widths[0], self.room_depths[0]))

                # Plot the rate map in the corresponding subplot
                title = f"Cell {j+1}"
                make_plot_rate_map(rate_map_mat, axs[ax_row, ax_col], title, "", "", "")

            # Hide unused subplots for the current frequency
            for j in range(n_cells, num_rows * num_cols):
                ax_row = j // num_cols
                ax_col = j % num_cols
                axs[ax_row, ax_col].axis("off")

            # Add a single colorbar to the figure
            cbar = fig.colorbar(axs[0, 0].get_images()[0], cax=cbar_ax)
            cbar.set_label("Firing rate", fontsize=14)

            figs.append(fig)
            axes.append(axs)

        return figs, axes
