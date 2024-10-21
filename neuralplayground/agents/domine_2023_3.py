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
from neuralplayground.agents.agent_core import AgentCore
from neuralplayground.agents.domine_2023_extras_2.utils.plotting_utils import plot_curves, plot_2dgraphs
from neuralplayground.agents.domine_2023_extras_2.models.GCN_model import GCNModel
from neuralplayground.agents.domine_2023_extras_2.class_grid_run_config import GridConfig
from neuralplayground.agents.domine_2023_extras_2.utils.utils import set_device
from neuralplayground.agents.domine_2023_extras_2.processing.Graph_generation import sample_random_graph, sample_target, sample_omniglot_graph
from torchmetrics import Accuracy, Precision, AUROC, Recall, MatthewsCorrCoef
from torchmetrics.classification import BinaryAccuracy
# from neuralplayground.agents.domine_2023_extras_2.evaluate import Evaluator
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class Domine2023(AgentCore):
    def __init__(self, experiment_name="smaller size generalisation graph with no position feature",
                 train_on_shortest_path=True, resample=True, wandb_on=False, seed=41, dataset = 'random',
                 weighted=True, num_hidden=100, num_layers=2, num_message_passing_steps=3, learning_rate=0.001,
                 num_training_steps=10, residual=True, layer_norm=True, batch_size=4, num_features=4, num_nodes_max=7,
                 batch_size_test=4, num_nodes_min_test=4, num_nodes_max_test=7, plot=True, **mod_kwargs):
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
        self.dataset = dataset
        self.weighted = weighted
        self.num_features = num_features
        self.num_nodes_max = num_nodes_max
        self.num_nodes_max_test = num_nodes_max_test

        self.batch_size_test = batch_size_test
        self.arena_x_limits = mod_kwargs["arena_x_limits"]
        self.arena_y_limits = mod_kwargs["arena_y_limits"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.dataset == 'random':
            self.model = GCNModel(self.num_hidden, self.num_features + 2, self.num_layers,
                                  self.num_message_passing_steps, self.residual,
                                  self.layer_norm).to(self.device)
        else:
            num_features = 784
            self.model = GCNModel(self.num_hidden, num_features + 2, self.num_layers,
                                  self.num_message_passing_steps, self.residual,
                                  self.layer_norm).to(self.device)


        self.auroc = AUROC(task="binary")
        self.MCC = MatthewsCorrCoef(task='binary')
        self.metric = BinaryAccuracy()


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
        self.wandb_logs = { # This is thought of the state density
            "batch_size": self.batch_size,
            "num_node_min": self.num_nodes_max,  # This is thought of the state density
            "seed": self.seed,
            "dataset": self.dataset,
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

    def load_data(self, train, dataset, batch_size):
        # Initialize lists to store batch data
        node_features_batch, edges_batch, edge_features_batch, target_batch = [], [], [], []

        # Determine the max nodes to use based on whether it is training or testing
        num_nodes = self.num_nodes_max if train else self.num_nodes_max_test

        # Loop to generate a batch of data
        for _ in range(batch_size):
            # Handle Omniglot dataset
            if dataset == 'omniglot':
                node_features, edges, edge_features_tensor, source, sink = sample_omniglot_graph(num_nodes)
            # Handle Random dataset
            elif dataset == 'random':
                node_features, edges, edge_features_tensor, source, sink = sample_random_graph(self.num_features,
                                                                                               num_nodes)

            # Sample the target based on source and sink
            target = sample_target(source, sink)

            # Append each graph data to the batch list
            node_features_batch.append(node_features)
            edges_batch.append(edges)
            edge_features_batch.append(edge_features_tensor)
            target_batch.append(target)

        return node_features_batch, edges_batch, edge_features_batch, target_batch

    def compute_loss(self, outputs, targets):
            loss = self.criterion(outputs, targets)
            return loss

    def run_model(self, node, edges,edges_features):
        outputs = self.model(node,edges,edges_features)
        return outputs

    def update_step(self, node_batch, edges_batch, edges_features_batch, target_batch, train, batch_size=1):
        if train:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        batch_losses = 0
        all_outputs = []

        # Loop over the batch
        for i in range(batch_size):
            data = node_batch[i].to(self.device)
            edges = edges_batch[i].to(self.device)
            edges_features = edges_features_batch[i].to(self.device)

            # Forward pass
            outputs = self.run_model(data, edges, edges_features)
            all_outputs.append(outputs)

            # Compute loss for this sample
            loss = self.compute_loss(outputs, target_batch[i])
            batch_losses += loss

        # Average loss over the batch
        avg_loss = batch_losses / batch_size

        if train:
            avg_loss.backward()
            self.optimizer.step()

        # Concatenate all outputs for evaluation over the batch
        all_outputs = torch.cat(all_outputs)
        all_target = torch.cat(target_batch)

        # Evaluate using the full batch's predictions
        roc_auc, mcc = self.evaluate(all_outputs, all_target)

        return avg_loss.item(), roc_auc, mcc

    def evaluate(self,outputs,targets):
            with (torch.no_grad()):
                roc_auc =  self.auroc(outputs, targets)
                # roc_auc_score(targets.cpu(), outputs.cpu())
                mcc =  self.MCC(outputs, targets)
                acc = self.metric(outputs, targets)
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
        node_features_val, edges_val, edge_features_tensor_val, target_val = self.load_data(train=False,
                                                                                            dataset=self.dataset,
                                                                                            batch_size=self.batch_size)

        for epoch in range(self.num_training_steps):
            train_losses, train_roc_auc, train_mcc = 0, 0, 0
            node_features, edges, edge_features_tensor, target = self.load_data(train=True,
                                                                                dataset=self.dataset,
                                                                                batch_size=self.batch_size)
            #Train on each batch
            batch_losses, batch_roc_auc, batch_mcc = self.update_step(node_features, edges,
                                                                          edge_features_tensor, target, train=True,  batch_size=self.batch_size)

            #Aggregate batch results
            train_losses = batch_losses
            train_roc_auc = batch_roc_auc.detach().numpy()
            train_mcc = batch_mcc.detach().numpy()
            # Average batch results over the batch size

            # Store results for plotting
            self.losses_train.append(train_losses)
            self.roc_aucs_train.append(train_roc_auc)
            self.MCCs_train.append(train_mcc)
            with torch.no_grad():
                val_losses, val_roc_auc, val_mcc = self.update_step(node_features_val, edges_val,
                                                                    edge_features_tensor_val, target_val,
                                                                    train=False,batch_size=self.batch_size)
            self.losses_val.append(val_losses)
            self.roc_aucs_val.append(val_roc_auc.detach().numpy())
            self.MCCs_val.append(val_mcc)

            # Log training details
            self.log_training(train_losses, val_losses, train_roc_auc,
                              val_roc_auc.detach().numpy(), train_mcc, val_mcc)

            # Plot progress every epoch
            if self.global_steps % self.log_every == 0:
                print(
                    f"Epoch {epoch}: Train Loss = {train_losses}, Val Loss = {val_losses}, ROC AUC Train = {train_roc_auc}, ROC AUC Val = {val_roc_auc}")
            self.global_steps += 1
        print("Finished training")

        if self.plot:
            os.mkdir(os.path.join(self.save_path, "results"))
            self.save_path = os.path.join(self.save_path, "results")
            plot_curves(
                [
                    self.losses_train,
                    self.losses_val],
                os.path.join(self.save_path, "Losses.pdf"),
                "All_Losses",
                legend_labels=["loss", "loss tesft"],
            )
            plot_curves(
                [
                    self.MCCs_train,
                ],
                os.path.join(self.save_path, "MCCs_train.pdf"),
                "MCC Train",
                legend_labels=["MCC Train"],
            )
            plot_curves(
                [
                    self.roc_aucs_train,
                ],
                os.path.join(self.save_path, "AUROC_train.pdf"),
                "AUROC Train",
                legend_labels=["AUROC Train"],
            )
            plot_curves(
                [
                    self.MCCs_val,
                ],
                os.path.join(self.save_path, "MCCs_val.pdf"),
                "MCC val",
                legend_labels=["MCC val"],
            )

        def sample_and_store(n):
            # Initialize empty lists to store each sample's output
            node_features_list = []
            edges_list = []
            edge_features_tensor_list = []
            target_list = []
            # Loop n times to sample data and store the outputs
            for _ in range(n):
                # Sample data by calling load_data
                node_features, edges, edge_features_tensor, target = self.load_data(train=True,dataset = self.dataset)
                # Append the results to the corresponding lists
                node_features_list.append(node_features)
                edges_list.append(edges)
                edge_features_tensor_list.append(edge_features_tensor)
                target_list.append(target)
            return node_features_list, edges_list, edge_features_tensor_list, target_list

        n=2
        node_features_list, edges_list, edge_features_tensor_list, target_list =  sample_and_store(n)
        plot_2dgraphs(edges_list, node_features_list, edge_features_tensor_list,['',''], os.path.join(self.save_path, "graph.pdf"), colorscale='Plasma',size=5,show=True)
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
        resample=config.resample,
        wandb_on=config.wandb_on,
        seed=config.seed,
        dataset=config.dataset,
        weighted=config.weighted,
        num_hidden=config.num_hidden,
        num_layers=config.num_layers,
        num_message_passing_steps=config.num_message_passing_steps,
        learning_rate=config.learning_rate,
        num_training_steps=config.num_training_steps,
        batch_size=config.batch_size,
        num_features=config.num_features,
        num_nodes_max=config.num_nodes_max,
        num_nodes_min=config.num_nodes_min,
        batch_size_test=config.batch_size_test,
        num_nodes_min_test=config.num_nodes_min_test,
        num_nodes_max_test=config.num_nodes_max_test,
        arena_y_limits=arena_y_limits,
        arena_x_limits=arena_x_limits,
        residual=config.residual,
        layer_norm=config.layer_norm,
        plot=config.plot,
    )

    agent.train()

    #TO DO : figure out how to build the graph and the task in that setting, will it be a batch of multople graphs, how to i compute the loss on asingle param?? Global ??
    # I need to check the saving and the logging of the results