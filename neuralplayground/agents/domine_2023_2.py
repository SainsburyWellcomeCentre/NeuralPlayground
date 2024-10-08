import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from datetime import datetime
import torch
import shutil
from datetime import datetime
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from neuralplayground.agents.agent_core import AgentCore
from neuralplayground.agents.domine_2023_extras_2.utils.plotting_utils import plot_curves
from neuralplayground.agents.domine_2023_extras_2.models.GCN_model import GCNModel
from neuralplayground.agents.domine_2023_extras_2.class_grid_run_config import GridConfig
from neuralplayground.agents.domine_2023_extras_2.utils.utils import set_device
from neuralplayground.agents.domine_2023_extras_2.processing.Graph_generation import sample_graph
from torchmetrics import Accuracy, Precision, AUROC, Recall, MatthewsCorrCoef
# from neuralplayground.agents.domine_2023_extras_2.evaluate import Evaluator
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class Domine2023(AgentCore):
    def __init__(self, experiment_name="smaller size generalisation graph with no position feature",
                 train_on_shortest_path=True, resample=True, wandb_on=False, seed=41, feature_position=False,
                 weighted=True, num_hidden=100, num_layers=2, num_message_passing_steps=3, learning_rate=0.001,
                 num_training_steps=10, residual=True, layer_norm=True, batch_size=4, nx_min=4, nx_max=7,
                 batch_size_test=4, nx_min_test=4, nx_max_test=7, plot=True, **mod_kwargs):
        super(Domine2023, self).__init__()

        # General
        self.plot = plot
        self.obs_history = []
        self.grad_history = []
        self.experiment_name = experiment_name
        self.wandb_on = wandb_on
        self.seed = seed
        self.log_every = 500

        # Network
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.num_message_passing_steps = num_message_passing_steps
        self.learning_rate = learning_rate
        self.num_training_steps = num_training_steps
        self.batch_size = batch_size
        self.residual = residual
        self.layer_norm = layer_norm

        # Task
        #self.resample = resample
        self.feature_position = feature_position
        self.weighted = weighted
        self.nx_min = nx_min
        self.nx_max = nx_max
        self.batch_size_test = batch_size_test
        self.nx_min_test = nx_min_test
        self.nx_max_test = nx_max_test
        self.arena_x_limits = mod_kwargs["arena_x_limits"]
        self.arena_y_limits = mod_kwargs["arena_y_limits"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GCNModel(self.num_hidden, self.num_layers, self.num_message_passing_steps, self.residual,
                              self.layer_norm).to(self.device)
        self.auroc = AUROC(task="binary")
        self.MCC = MatthewsCorrCoef(task='binary')
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        if self.wandb_on:
            dateTimeObj = datetime.now()
            wandb.init(project="New", entity="graph-brain",
                       name=experiment_name + dateTimeObj.strftime("%d%b_%H_%M_%S"))
            self.wandb_logs = {}
            save_path = wandb.run.dir
            os.mkdir(os.path.join(save_path,"results"))
            self.save_path = os.path.join(save_path, "results")

        self.reset()
        self.wandb_logs = {
            "nx_min_test": self.nx_min_test,  # This is thought of the state density
            "nx_max_test": self.nx_max_test,  # This is thought of the state density
            "batch_size": self.batch_size,
            "nx_min": self.nx_min,  # This is thought of the state density
            "nx_max": self.nx_max,
            "seed": self.seed,
            "feature_position": self.feature_position,
            "weighted": self.weighted,
            "num_hidden": self.num_hidden,
            "num_layers": self.num_layers,
            "num_message_passing_steps": self.num_message_passing_steps,
            "learning_rate": self.learning_rate,
            "num_training_steps": self.num_training_steps,
            "residual": self.residual,
            "layer_norm": self.layer_norm,
        }

        if self.wandb_on:
            wandb.log(self.wandb_logs)
        else:
            dateTimeObj = datetime.now()
            save_path = os.path.join(Path(os.getcwd()).resolve(), "results")
            os.mkdir(
                os.path.join(
                    save_path,
                    self.experiment_name + dateTimeObj.strftime("%d%b_%H_%M_%S"),
                )
            )
            self.save_path = os.path.join(
                os.path.join(
                    save_path,
                    self.experiment_name + dateTimeObj.strftime("%d%b_%H_%M_%S"),
                )
            )
        self.save_run_parameters()

    def save_run_parameters(self):
        """Save configuration files and scripts."""
        files_to_copy = [
            ("run.py", "domine_2023_2.py"),
            ("Graph_generation.py", "domine_2023_extras_2/processing/Graph_generation.py"),
            ("utils.py", "domine_2023_extras_2/utils/utils.py"),
            ("plotting_utils.py", "domine_2023_extras_2/utils/plotting_utils.py"),
            ("config_run.yaml", "domine_2023_extras_2/config.yaml"),
        ]
        for file_name, source in files_to_copy:
            shutil.copyfile(os.path.join(Path(os.getcwd()).resolve(), source), os.path.join(self.save_path, file_name))


    def load_data(self,train):
        if train:
            node, adj = sample_graph(train=True)
        else:
            node, adj = sample_graph(train=False)
        return node, adj


    def compute_loss(self, outputs, targets):
            loss = self.criterion(outputs, targets)
            return loss

    def run_model(self, node, edges):
        outputs = self.model(node,edges)
        return outputs

    def update_step(self,node, edges,target,train):
        data = node.to(self.device)
        edges = edges.to(self.device)
        if train:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()
        outputs = self.run_model(data,edges)
        loss = self.compute_loss(outputs,target)
        if train:
            loss.backward()
            self.optimizer.step()
        roc_auc, mcc = self.evaluate(outputs,target)
        return loss,roc_auc, mcc

    def evaluate(self,outputs,targets):
            with (torch.no_grad()):
                roc_auc =  self.auroc(outputs.to(self.device), targets.to(self.device))
                # roc_auc_score(targets.cpu(), outputs.cpu())
                # mcc = MatthewsCorrCoef(outputs.cpu().round(), targets.cpu().round())
                mcc = 1
            return roc_auc, mcc

    def log_training(self, train_loss, val_loss, train_roc_auc, val_roc_auc, train_mcc, val_mcc):
        """Log training and validation metrics."""
        wandb_logs = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "roc_auc_train": train_roc_auc,
            "roc_auc_val": val_roc_auc,
            "MCC_train": train_mcc,
            "MCC_val": val_mcc
        }
        if self.wandb_on:
            wandb.log(wandb_logs)

    def train(self):
        for epoch in range(self.num_training_steps):

            nodes, edges = self.load_data(train=True)
            target = nodes
            train_losses, train_roc_auc, train_mcc = self.update_step(nodes, edges, target ,train=True)
            self.losses_train.append(train_losses.detach().numpy() )
            self.MCCs_train.append(train_mcc)
            self.roc_aucs_train.append(train_roc_auc.detach().numpy() )
            nodes_val, edges_val = self.load_data(train=False)
            with torch.no_grad():
                val_losses, val_roc_auc, val_mcc = self.update_step(nodes_val,edges_val,target, train=False)
            self.losses_val.append(val_losses.detach().numpy() )
            self.MCCs_val.append(val_mcc)
            self.roc_aucs_val.append(val_roc_auc.detach().numpy() )
            self.log_training(train_losses.detach().numpy(), val_losses.detach().numpy(), train_roc_auc.detach().numpy(), val_roc_auc.detach().numpy(), train_mcc, val_mcc)
            self.global_steps = self.global_steps + 1
            if self.global_steps % self.log_every == 0:
                print(
                    f"Training step {self.global_steps}: log_loss = {np.log(train_losses.detach().numpy() )} , log_loss_test = {np.log(val_losses.detach().numpy() )}, roc_auc_test = {val_roc_auc}, roc_auc_train = {train_roc_auc}"
                )
        print("Finished training")
        if self.plot:
            plot_curves(
                [
                    self.losses_train,
                    self.losses_val],
                os.path.join(self.save_path, "Losses.pdf"),
                "All_Losses",
                legend_labels=["loss", "loss test"],
            )
            plot_curves(
                [
                    self.MCCs_train,
                ],
                os.path.join(self.save_path, "MCCs_train.pdf"),
                "All_Losses",
                legend_labels=["MCC Train"],
            )
            plot_curves(
                [
                    self.roc_aucs_train,
                ],
                os.path.join(self.save_path, "MCCs_train.pdf"),
                "All_Losses",
                legend_labels=["MCC Train"],
            )
            plot_curves(
                [
                    self.MCCs_val,
                ],
                os.path.join(self.save_path, "MCCs_train.pdf"),
                "All_Losses",
                legend_labels=["MCC Train"],
            )
            plot_curves(
                [
                    self.roc_aucs_train,
                ],
                os.path.join(self.save_path, "MCCs_train.pdf"),
                "All_Losses",
                legend_labels=["MCC Train"],
            )
        return

    def reset(self):
        self.obs_history = []
        self.grad_history = []
        self.global_steps = 0
        self.losses_train = []
        self.losses_val = []
        self.MCCs_train = []
        self.MCCs_val = []
        self.roc_aucs_train = []
        self.roc_aucs_val = []
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", metavar="-C", default="domine_2023_extras_2/config.yaml",
                        help="path to base configuration file.")
    args = parser.parse_args()
    set_device()
    config_class = GridConfig
    config = config_class(args.config_path)

    arena_x_limits = [-100, 100]
    arena_y_limits = [-100, 100]

    agent = Domine2023(
        experiment_name=config.experiment_name,
        train_on_shortest_path=config.train_on_shortest_path,
        resample=config.resample,
        wandb_on=config.wandb_on,
        seed=config.seed,
        feature_position=config.feature_position,
        weighted=config.weighted,
        num_hidden=config.num_hidden,
        num_layers=config.num_layers,
        num_message_passing_steps=config.num_message_passing_steps,
        learning_rate=config.learning_rate,
        num_training_steps=config.num_training_steps,
        batch_size=config.batch_size,
        nx_min=config.nx_min,
        nx_max=config.nx_max,
        batch_size_test=config.batch_size_test,
        nx_min_test=config.nx_min_test,
        nx_max_test=config.nx_max_test,
        arena_y_limits=arena_y_limits,
        arena_x_limits=arena_x_limits,
        residual=config.residual,
        layer_norm=config.layer_norm,
        grid=config.grid,
        plot=config.plot,
        dist_cutoff=config.dist_cutoff,
        n_std_dist_cutoff=config.n_std_dist_cutoff,
    )

    agent.train()

    #TO DO : figure out how to build the graph and the task in that setting, will it be a batch of multople graphs, how to i compute the loss on asingle param?? Global ??
    # I need to check the saving and the logging of the results