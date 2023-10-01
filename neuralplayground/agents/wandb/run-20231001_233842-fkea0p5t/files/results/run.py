import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from agent_core import AgentCore

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from class_Graph_generation import sample_padded_grid_batch_shortest_path
from class_grid_run_config import GridConfig
from class_models import get_forward_function
from class_plotting_utils import (
    plot_graph_grid_activations,
    plot_input_target_output,
    plot_message_passing_layers,
    plot_xy,
)
from class_utils import rng_sequence_from_rng, set_device
from sklearn.metrics import matthews_corrcoef, roc_auc_score

# @title Graph net functions
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path",
    metavar="-C",
    default="class_config.yaml",
    help="path to base configuration file.",
)


class Stachenfel2023(AgentCore):
    def __init__(self, config_path, config, **mod_kwargs):
        self.train_on_shortest_path = config.train_on_shortest_path
        # @param
        super().__init__()
        self.experiment_name = config.experiment_name
        self.train_on_shortest_path = config.train_on_shortest_path
        self.resample = config.resample  # @param
        self.wandb_on = config.wandb_on
        self.seed = config.seed

        self.feature_position = config.feature_position
        self.weighted = config.weighted

        self.num_hidden = config.num_hidden  # @param
        self.num_layers = config.num_layers  # @param
        self.num_message_passing_steps = config.num_training_steps  # @param
        self.learning_rate = config.learning_rate  # @param
        self.num_training_steps = config.num_training_steps  # @param

        self.batch_size = config.batch_size
        self.nx_min = config.nx_min
        self.nx_max = config.nx_max
        self.arena_x_limits = mod_kwargs["arena_x_limits"]
        self.arena_y_limits = mod_kwargs["arena_y_limits"]
        self.room_width = np.diff(self.arena_x_limits)[0]
        self.room_depth = np.diff(self.arena_y_limits)[0]

        # This can be tought of the brain making different rep of different  granularity
        # Could be explained during sleep
        self.batch_size_test = config.batch_size_test
        self.nx_min_test = config.nx_min_test  # This is thought of the state density
        self.nx_max_test = config.nx_max_test  # This is thought of the state density
        self.batch_size = config.batch_size
        self.nx_min = config.nx_min  # This is thought of the state density
        self.nx_max = config.nx_max  # This is thought of the state density

        # TODO: Make sure that for different graph this changes with the environement
        # self.ny_min_test = config.ny_min_test  # This is thought of the state density
        # self.ny_max_test = config.ny_max_test  # This is thought of the state density
        # self.ny_min = con
        # fig.ny_min  # This is thought of the state density
        # self.ny_max = config.ny_max  # This is thought of the state density

        # self.resolution_x_min_test = int(self.nx_min * self.room_width)
        # self.resolution_x_max_test = int(self.nx_max * self.room_depth)
        # self.resolution_x_min = int(self.nx_min_test * self.room_width)
        # self.resolution_x_max = int(self.nx_max_test * self.room_depth)

        # self.resolution_y_min_test = int(self.nx_min * self.room_width)
        # self.resolution_y_max_test = int(self.nx_max * self.room_depth)
        # self.resolution_y_min = int(self.nx_min_test * self.room_width)
        # self.resolution_y_max = int(self.nx_max_test * self.room_depth)

        self.log_every = config.num_training_steps // 10

        if self.weighted:
            self.edege_lables = True
        else:
            self.edege_lables = False
        if self.wandb_on:
            dateTimeObj = datetime.now()
            wandb.init(
                project="graph-brain", entity="graph-brain", name="Grid_shortest_path" + dateTimeObj.strftime("%d%b_%H_%M")
            )
            self.wandb_logs = {}
            save_path = wandb.run.dir
            os.mkdir(os.path.join(save_path, "results"))
            self.save_path = os.path.join(save_path, "results")
        else:
            dateTimeObj = datetime.now()
            save_path = os.path.join(Path(os.getcwd()).resolve(), "results")
            os.mkdir(os.path.join(save_path, "Grid_shortest_path" + dateTimeObj.strftime("%d%b_%H_%M")))
            self.save_path = os.path.join(os.path.join(save_path, "Grid_shortest_path" + dateTimeObj.strftime("%d%b_%H_%M")))

        # SAVING Trainning Files
        path = os.path.join(self.save_path, "run.py")
        HERE = os.path.join(Path(os.getcwd()).resolve(), "class.py")
        shutil.copyfile(HERE, path)

        path = os.path.join(self.save_path, "class_Graph_generation.py")
        HERE = os.path.join(Path(os.getcwd()).resolve(), "class_Graph_generation.py")
        shutil.copyfile(HERE, path)

        path = os.path.join(self.save_path, "class_utils.py")
        HERE = os.path.join(Path(os.getcwd()).resolve(), "class_utils.py")
        shutil.copyfile(HERE, path)

        path = os.path.join(self.save_path, "class_plotting_utils.py")
        HERE = os.path.join(Path(os.getcwd()).resolve(), "class_plotting_utils.py")
        shutil.copyfile(HERE, path)

        path = os.path.join(self.save_path, "class_config_run.yaml")
        HERE = os.path.join(Path(os.getcwd()).resolve(), "class_config.yaml")
        shutil.copyfile(HERE, path)

        # This is the function that does the forward pass of the model
        forward = get_forward_function(self.num_hidden, self.num_layers, self.num_message_passing_steps)
        self.net_hk = hk.without_apply_rng(hk.transform(forward))
        self.rng = jax.random.PRNGKey(self.seed)
        self.rng_seq = rng_sequence_from_rng(self.rng)

        if self.train_on_shortest_path:
            graph, targets = sample_padded_grid_batch_shortest_path(
                self.rng, self.batch_size, self.feature_position, self.weighted, self.nx_min, self.nx_max
            )
        else:
            graph, targets = sample_padded_grid_batch_shortest_path(
                self.rng, self.batch_size, self.feature_position, self.weighted, self.nx_min, self.nx_max
            )
        self.params = self.net_hk.init(self.rng, graph)
        self.optimizer = optax.adam(self.learning_rate)
        self.opt_state = self.optimizer.init(self.params)

        self.reset()

    def reset(self):
        # TODO: Actually reset the network
        self.global_test = 0
        self.losses = []
        self.losses_test = []
        self.roc_aucs_train = []
        self.MCCs_train = []
        self.MCCs_test = []
        self.roc_aucs_test = []

    @jax.jit
    def compute_loss(self, params, model, inputs, targets):
        # not jitted because it will get jitted in jax.value_and_grad
        outputs = model.apply(params, inputs)
        return jnp.mean((outputs[0].nodes - targets) ** 2)  # using MSE

    @jax.jit
    def update_step(self, grads, opt_state, params, optimizer):
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params

    @jax.jit
    def evaluate(self, model, params, inputs, target):
        outputs = model.apply(params, inputs)
        roc_auc = roc_auc_score(np.squeeze(target), np.squeeze(outputs[0].nodes))
        MCC = matthews_corrcoef(np.squeeze(target), round(np.squeeze(outputs[0].nodes)))

        return outputs, roc_auc, MCC

    def update(self):
        rng = next(self.rng_seq)
        graph_test, target_test = sample_padded_grid_batch_shortest_path(
            rng, self.batch_size_test, self.feature_position, self.weighted, self.nx_min_test, self.nx_max_test
        )
        rng = next(self.rng_seq)
        # Sample a new batch of graph every itterations
        if self.resample:
            if self.train_on_shortest_path:
                graph, targets = sample_padded_grid_batch_shortest_path(
                    rng, self.batch_size, self.feature_position, self.weighted, self.nx_min, self.nx_max
                )
            else:
                graph, targets = sample_padded_grid_batch_shortest_path(
                    rng, self.batch_size, self.feature_position, self.weighted, self.nx_min, self.nx_max
                )
                targets = graph.nodes
        # Train
        loss, grads = jax.value_and_grad(self.compute_loss)(
            self.params, self.net_hk, graph, targets
        )  # jits inside of value_and_grad
        self.params = self.update_step(grads, self.opt_state, self.params, self.optimizer)
        self.losses.append(loss)
        outputs_train, roc_auc_train, MCC_train = self.evaluate(self.net_hk, self.params, graph, targets)
        self.roc_aucs_train.append(roc_auc_train)
        self.MCCs_train.append(MCC_train)  # Matthews correlation coefficient
        # Test # model should basically learn to do nothing from this
        loss_test = self.compute_loss(self.params, self.net_hk, graph_test, target_test)
        self.losses_test.append(loss_test)
        outputs_test, roc_auc_test, MCC_test = self.evaluate(self.net_hk, self.params, graph_test, target_test)
        self.roc_aucs_test.append(roc_auc_test)
        self.MCCs_test.append(MCC_test)

        # Log
        wandb_logs = {"loss": loss, "losses_test": loss_test, "roc_auc_test": roc_auc_test, "roc_auc": roc_auc_train}
        if self.wandb_on:
            wandb.log(wandb_logs)
        self.global_steps = self.global_steps + 1
        if self.global_steps % self.log_every == 0:
            print(f"Training step {n}: loss = {loss}")
        return

    def print_and_plot(self):
        # EVALUATE
        rng = next(self.rng_seq)
        graph_test, target_test = sample_padded_grid_batch_shortest_path(
            rng, self.batch_size_test, self.feature_position, self.weighted, self.nx_min_test, self.nx_max_test
        )
        outputs, roc_auc, MCC = self.evaluate(self.net_hk, self.params, graph_test, target_test)
        print("roc_auc_score")
        print(roc_auc)
        print("MCC")
        print(MCC)

        # SAVE PARAMETER (NOT WE SAVE THE FILES SO IT SHOULD BE THERE AS WELL )
        if self.wandb_on:
            with open("readme.txt", "w") as f:
                f.write("readme")
            with open(os.path.join(self.save_path, "Constant.txt"), "w") as outfile:
                outfile.write("num_message_passing_steps" + str(self.num_message_passing_steps) + "\n")
                outfile.write("Learning_rate:" + str(self.learning_rate) + "\n")
                outfile.write("num_training_steps:" + str(self.num_training_steps))
                outfile.write("roc_auc" + str(roc_auc))
                outfile.write("MCC" + str(MCC))

        # PLOTTING THE LOSS and AUC ROC
        plot_xy(self.losses, os.path.join(self.save_path, "Losses.pdf"), "Losses")
        plot_xy(self.losses_test, os.path.join(self.save_path, "Losses_test.pdf"), "Losses_test")
        plot_xy(self.roc_aucs_test, os.path.join(self.save_path, "auc_roc_test.pdf"), "auc_roc_test")
        plot_xy(self.roc_aucs_train, os.path.join(self.save_path, "auc_roc_train.pdf"), "auc_roc_train")
        plot_xy(self.MCCs_train, os.path.join(self.save_path, "MCC_train.pdf"), "MCC_train")
        plot_xy(self.MCCs_test, os.path.join(self.save_path, "MCC_test.pdf"), "MCC_test")

        # PLOTTING ACTIVATION OF THE FIRST 2 GRAPH OF THE BATCH
        plot_input_target_output(
            list(graph_test.nodes.sum(-1)),
            target_test.sum(-1),
            outputs[0].nodes.tolist(),
            graph_test,
            4,
            self.edege_lables,
            os.path.join(self.save_path, "in_out_targ.pdf"),
        )
        plot_message_passing_layers(
            list(graph_test.nodes.sum(-1)),
            outputs[1],
            target_test.sum(-1),
            outputs[0].nodes.tolist(),
            graph_test,
            3,
            self.num_message_passing_steps,
            self.edege_lables,
            os.path.join(self.save_path, "message_passing_graph.pdf"),
        )
        # plot_message_passing_layers_units(outputs[1], target_test.sum(-1), outputs[0].nodes.tolist(),graph_test,config.num_hidden,config.num_message_passing_steps,edege_lables,os.path.join(save_path, 'message_passing_hidden_unit.pdf'))

        # Plot each seperatly
        plot_graph_grid_activations(
            outputs[0].nodes.tolist(),
            graph_test,
            os.path.join(self.save_path, "outputs.pdf"),
            "Predicted Node Assignments with GCN",
            self.edege_lables,
        )
        plot_graph_grid_activations(
            list(graph_test.nodes.sum(-1)),
            graph_test,
            os.path.join(self.save_path, "Inputs.pdf"),
            "Inputs node assigments",
            self.edege_lables,
        )
        plot_graph_grid_activations(
            target_test.sum(-1), graph_test, os.path.join(self.save_path, "Target.pdf"), "Target", self.edege_lables
        )

        plot_graph_grid_activations(
            outputs[0].nodes.tolist(),
            graph_test,
            os.path.join(self.save_path, "outputs_2.pdf"),
            "Predicted Node Assignments with GCN",
            self.edege_lables,
            2,
        )
        plot_graph_grid_activations(
            list(graph_test.nodes.sum(-1)),
            graph_test,
            os.path.join(self.save_path, "Inputs_2.pdf"),
            "Inputs node assigments",
            self.edege_lables,
            2,
        )
        plot_graph_grid_activations(
            target_test.sum(-1), graph_test, os.path.join(self.save_path, "Target_2.pdf"), "Target", self.edege_lables, 2
        )
        return


if __name__ == "__main__":
    from neuralplayground.arenas import Simple2D

    args = parser.parse_args()
    set_device()
    config_class = GridConfig
    config = config_class(args.config_path)
    time_step_size = 0.1  # seg
    agent_step_size = 3

    # Init environment
    arena_x_limits = [-100, 100]
    arena_y_limits = [-100, 100]
    env = Simple2D(
        time_step_size=time_step_size,
        agent_step_size=agent_step_size,
        arena_x_limits=arena_x_limits,
        arena_y_limits=arena_y_limits,
    )

    agent = Stachenfel2023(
        config_path=args.config_path, config=config, arena_y_limits=arena_y_limits, arena_x_limits=arena_x_limits
    )
    for n in range(config.num_training_steps):
        agent.update()

    agent.print_and_plot()


# TODO: Make it work with the config Run manadger
# TODO: Make juste an env type
# TODO: Make The plotting in the general plotting utilse
