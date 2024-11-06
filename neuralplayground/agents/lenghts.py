import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import torch
import shutil
from datetime import datetime
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
from neuralplayground.agents.agent_core import AgentCore
from neuralplayground.agents.domine_2023_extras_2.utils.plotting_utils import plot_curves, plot_curves_2, plot_2dgraphs
from neuralplayground.agents.domine_2023_extras_2.models.GCN_model import GCNModel, MLP ,GCNModel_2
from neuralplayground.agents.domine_2023_extras_2.class_grid_run_config import GridConfig
from neuralplayground.agents.domine_2023_extras_2.utils.utils import set_device
from neuralplayground.agents.domine_2023_extras_2.processing.Graph_generation import sample_graph, sample_target, sample_omniglot_graph, sample_fixed_graph
from torchmetrics import Accuracy, Precision, AUROC, Recall, MatthewsCorrCoef
from torchmetrics.classification import BinaryAccuracy

# from neuralplayground.agents.domine_2023_extras_2.evaluate import Evaluator
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

class Domine2023(AgentCore):
    def __init__(self, experiment_name="smaller size generalisation graph with no position feature",
                 train_on_shortest_path=True,  wandb_on=False, seed=41, dataset = 'random',
                  num_hidden=100, num_layers=2, num_message_passing_steps=3, learning_rate=0.001,
                 num_training_steps=10, residual=True, batch_size=4, num_features=4, num_nodes_max=7,
                 batch_size_test=4, num_nodes_min_test=4, num_nodes_max_test=[7], plot=True,  **mod_kwargs):
        super(Domine2023, self).__init__()

        # General
        np.random.seed(seed)
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


        # Task
        self.dataset = dataset
        self.num_features = num_features
        self.num_nodes_max = num_nodes_max
        self.num_nodes_max_test = num_nodes_max_test


        def set_initial_seed(seed):
            # Set the seed for NumPy
            np.random.seed(seed)

            # Set the seed for PyTorch
            torch.manual_seed(seed)

            # If using CUDA
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU

            # Optional: For deterministic behavior, you can use the following:
            # This is usually needed for reproducibility with certain layers like convolution.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        set_initial_seed(seed)

        self.batch_size_test = batch_size_test
        self.arena_x_limits = mod_kwargs["arena_x_limits"]
        self.arena_y_limits = mod_kwargs["arena_y_limits"]
        save_path = mod_kwargs["save_path"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.dataset == 'random':
            self.model = GCNModel(self.num_hidden, self.num_features + 2, self.num_layers,
                                  self.num_message_passing_steps, self.residual
                                  ).to(self.device)
        elif self.dataset == 'positional':
            self.model = GCNModel(self.num_hidden, self.num_features + 3, self.num_layers,
                                  self.num_message_passing_steps, self.residual
                                  ).to(self.device)
        elif self.dataset == 'positional_no_edges':
            self.model = GCNModel_2(self.num_hidden, self.num_features + 3, self.num_layers,
                                  self.num_message_passing_steps, self.residual
                                 ).to(self.device)
        else:
            num_features = 784
            self.model = GCNModel(self.num_hidden, num_features + 2, self.num_layers,
                                  self.num_message_passing_steps, self.residual,
                                  ).to(self.device)


        self.auroc = AUROC(task="binary")
        self.ACC = BinaryAccuracy()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
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
            "num_hidden": self.num_hidden,
            "num_layers": self.num_layers,
            "num_message_passing_steps": self.num_message_passing_steps,
            "learning_rate": self.learning_rate,
            "num_training_steps": self.num_training_steps,
            "residual": self.residual,
        }
        if self.wandb_on:
            wandb.log(self.wandb_logs)
        else:
            self.save_path= save_path
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

    def load_data(self, fixed, dataset, batch_size,num_nodes):
        # Initialize lists to store batch data
        node_features_batch, edges_batch, edge_features_batch, target_batch = [], [], [], []

        # Determine the max nodes to use based on whether it is training or testing

        # Loop to generate a batch of data
        for _ in range(batch_size):
            if fixed:
                node_features, edges, edge_features_tensor, source, sink = sample_fixed_graph(self.num_features,
                                                                                              num_nodes,feature_type=  dataset)
            else:
            # Handle Omniglot dataset
                if dataset == 'omniglot':
                    node_features, edges, edge_features_tensor, source, sink = sample_omniglot_graph(num_nodes)
                # Handle Random dataset
                else:
                    node_features, edges, edge_features_tensor, source, sink = sample_graph(self.num_features,
                                                                                                   num_nodes,feature_type= dataset)



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
            all_outputs.append(outputs.view(-1))
        # Compute loss for this sample
            loss = self.compute_loss(outputs, target_batch[i])
            #target = torch.randn(2, 2).softmax(dim=1)
            #input = target * 0.99999
            #self.compute_loss(input, target)

            batch_losses += loss

        # Average loss over the batch
        avg_loss = batch_losses / batch_size
        #all_outputs = torch.stack(all_outputs)
       # all_target_1 = torch.stack(target_batch).view(-1)
        # all_target = torch.stack(target_batch)
        # loss = self.compute_loss(all_outputs, all_target_1)

        if train:
            avg_loss.backward()
            self.optimizer.step()

        # Concatenate all outputs for evaluation over the batch
        all_outputs = torch.stack(all_outputs)
        all_target = torch.stack(target_batch)
        # Evaluate using the full batch's predictions
        acc = self.evaluate(all_outputs, all_target)

        return avg_loss.item(), acc

    def evaluate(self,outputs,targets):
            with (torch.no_grad()):
                # roc_auc_score(targets.cpu(), outputs.cpu())
                labels = targets.view(-1)  # Outputs: [0, 1, 0, 1]
                # Convert predicted probabilities to class labels using argmax
                Softmax = nn.Softmax(dim=1)
                outputs = Softmax(outputs)
                predicted_labels = torch.argmax(outputs, dim=1)  # Outputs: [0, 1, 0, 1]
                # Initialize the BinaryAccuracy metric
                acc = self.ACC(predicted_labels, labels)
            return acc

    def log_training(self, train_loss, val_loss, train_acc, val_acc):
        """Log training and validation metrics."""
        wandb_logs = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "ACC_train": train_acc,
            "ACC_val": val_acc
        }
        if self.wandb_on:
            wandb.log(wandb_logs)

    def train(self):
        # Load validation data
        val_graphs = {num_node: [] for num_node in self.num_nodes_max_test}
        for i in range(len(self.num_nodes_max_test)):
            node_features_val, edges_val, edge_features_tensor_val, target_val = self.load_data(fixed=False,
                                                                                            dataset=self.dataset,
                                                                                             batch_size=self.batch_size, num_nodes= self.num_nodes_max_test[i])
            val_graphs[self.num_nodes_max_test[i]] = [node_features_val, edges_val, edge_features_tensor_val, target_val]

        #for i in len(self.num_nodes_max_test):
        # This is an attemp
        node_features_val_f, edges_val_f, edge_features_tensor_val_f, target_val_f = self.load_data(fixed=True,
                                                                                            dataset=self.dataset,
                                                                                            batch_size=self.batch_size,
                                                                                            num_nodes=
                                                                                            self.num_nodes_max_test[0],
                                                                                        )
         # need to save the fixed one node_featur

        for epoch in range(self.num_training_steps):

            node_features, edges, edge_features_tensor, target = self.load_data(fixed=False,
                                                                                dataset=self.dataset,
                                                                                batch_size=self.batch_size, num_nodes= self.num_nodes_max)
            #Train on each batch
            batch_losses, batch_acc = self.update_step(node_features, edges,
                                                                          edge_features_tensor, target, train=True,  batch_size=self.batch_size)

            #Aggregate batch results
            train_losses = batch_losses
            train_acc = batch_acc.detach().numpy()
            # Average batch results over the batch size

            # Store results for plotting
            self.losses_train.append(train_losses)

            self.ACCs_train.append(train_acc)

            # Validation
            with torch.no_grad():
                for num_node in self.num_nodes_max_test:
                    node_features_val, edges_val, edge_features_tensor_val, target_val = val_graphs[num_node]
                    val_losses, val_acc = self.update_step(node_features_val, edges_val,
                                                                    edge_features_tensor_val, target_val,
                                                                    train=False,batch_size=self.batch_size)
                    self.losses_val[num_node].append(val_losses)
                    self.ACCs_val[num_node].append(val_acc)


            # Log training details
            #TODO: Need to update this
            self.log_training(train_losses, val_losses, train_acc, val_acc)

            # Plot progress every epoch
            if self.global_steps % self.log_every == 0:
                print(
                    f"Epoch {epoch}: Train Loss = {train_losses}, Val Loss = {val_losses}, ACC Train = {train_acc}, ACC Val = {val_acc} ")
            self.global_steps += 1
        print("Finished training")


        if self.plot:
            os.makedirs(os.path.join(self.save_path, "results"), exist_ok=True)
            self.save_path = os.path.join(self.save_path, "results")
            file_name = f"Losses_{seed}.pdf"

            # Combine the path and file name
            list_of_lists = [value for value in self.losses_val.values()]
            list_of_list_name = [f'loss_val_len{value}' for value in self.losses_val]
            # Append losses_train to the list of lists
            list_of_lists.append(self.losses_train)
            list_of_list_name.append('loss_train')

            plot_curves(
                list_of_lists                ,
                os.path.join(self.save_path, file_name ),
                "All_Losses",
                legend_labels=list_of_list_name,
            )

            file_name = f"ACCs_val_{seed}.pdf"
            plot_curves( [value for value in self.ACCs_val.values()],
                os.path.join(self.save_path, file_name),
                "ACC Val",
                legend_labels=[f'ACC_val_len{value}' for value in self.losses_val],
            )
            file_name =  f"ACCs_train_{seed}.pdf"

            plot_curves(
                [
                    self.ACCs_train,
                ],
                os.path.join(self.save_path, file_name),
                "ACC train",
                legend_labels=["ACC train"],
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
                node_features, edges, edge_features_tensor, target = self.load_data(train=False,
                                                                                dataset=self.dataset,
                                                                                batch_size=self.batch_size)
                # Append the results to the corresponding lists
                node_features_list.append(node_features)
                edges_list.append(edges)
                edge_features_tensor_list.append(edge_features_tensor)
                target_list.append(target)
            return node_features_list, edges_list, edge_features_tensor_list, target_list
        n=2
        #node_features_list, edges_list, edge_features_tensor_list, target_list =  sample_and_store(n)
        #plot_2dgraphs(edges_list, node_features_list, edge_features_tensor_list,['',''], os.path.join(self.save_path, "graph.pdf"), colorscale='Plasma',size=5,show=True)
        return self.losses_train, self.ACCs_train, self.losses_val, self.ACCs_val
    def reset(self):
        self.obs_history = []
        self.grad_history = []
        self.global_steps = 0
        self.losses_train  = []
        self.losses_val = {num_node: [] for num_node in self.num_nodes_max_test}
        self.ACCs_val = {num_node: [] for num_node in self.num_nodes_max_test}
        self.ACCs_train = []

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

    seeds = [41,42]
    losses_train = {seed: [] for seed in seeds}
    losses_val = {seed: [] for seed in seeds}
    ACCs_train = {seed: [] for seed in seeds}
    ACCs_val = {seed: [] for seed in seeds}
    dateTimeObj = datetime.now()
    save_path = os.path.join(Path(os.getcwd()).resolve(), "results")
    os.mkdir(
        os.path.join(
            save_path,
            config.experiment_name + dateTimeObj.strftime("%d%b_%H_%M_%S"),
        )
    )
    save_path = os.path.join(
        os.path.join(
            save_path,
            config.experiment_name + dateTimeObj.strftime("%d%b_%H_%M_%S"),
        )
    )
    for seed in seeds:
        agent = Domine2023(
            experiment_name=config.experiment_name,
            wandb_on=config.wandb_on,
            seed=seed,
            dataset=config.dataset,
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
            plot=config.plot,
            save_path = save_path
        )



        losse_train, ACC_train, losse_val, ACC_val = agent.train()
        losses_train[seed] = losse_train
        losses_val[seed] = losse_val
        ACCs_train[seed]= ACC_train
        ACCs_val[seed] = ACC_val

    save_path = os.path.join(save_path, "results")

    num_training_steps = config.num_training_steps
    # Initialize lists to store standard deviation results
    std_losses_train = []
    std_accs_train = []

    # Compute average and standard deviation for training loss

    avg_losses_train = []
    for epoch_idx in range(num_training_steps):
        # Average the loss for this epoch over all seeds
        avg_epoch_loss = sum(losses_train[seed][epoch_idx] for seed in seeds) / len(seeds)
        avg_losses_train.append(avg_epoch_loss)

        # Compute standard deviation for this epoch
        variance_loss = sum((losses_train[seed][epoch_idx] - avg_epoch_loss) ** 2 for seed in seeds) / len(seeds)
        std_epoch_loss = math.sqrt(variance_loss)
        std_losses_train.append(std_epoch_loss)

    # Compute average and standard deviation for training accuracy
    avg_accs_train = []
    for epoch_idx in range(num_training_steps):
        # Average the accuracy for this epoch over all seeds
        avg_epoch_acc = sum(ACCs_train[seed][epoch_idx] for seed in seeds) / len(seeds)
        avg_accs_train.append(avg_epoch_acc)

        # Compute standard deviation for this epoch
        variance_acc = sum((ACCs_train[seed][epoch_idx] - avg_epoch_acc) ** 2 for seed in seeds) / len(seeds)
        std_epoch_acc = math.sqrt(variance_acc)
        std_accs_train.append(std_epoch_acc)


    # Compute average and standard deviation for validation loss
    avg_losses_val_len = []
    std_losses_val_len = []
    for i in config.num_nodes_max_test:
        avg_losses_val = []
        std_losses_val = []
        for epoch_idx in range(num_training_steps):
            avg_epoch_loss_val = sum(losses_val[seed][i][epoch_idx] for seed in seeds) / len(seeds)
            avg_losses_val.append(avg_epoch_loss_val)
            variance_loss_val = sum(
                (losses_val[seed][i][epoch_idx] - avg_epoch_loss_val) ** 2 for seed in seeds) / len(seeds)
            std_epoch_loss_val = math.sqrt(variance_loss_val)
            std_losses_val.append(std_epoch_loss_val)
        avg_losses_val_len.append(avg_losses_val)
        std_losses_val_len.append(std_losses_val)

    #Compute average and standard deviation for validation accuracy
    avg_accs_val_len = []
    std_accs_val_len = []
    for i in config.num_nodes_max_test:
        avg_accs_val = []
        std_accs_val = []
        for epoch_idx in range(num_training_steps):
            avg_epoch_acc_val = sum(ACCs_val[seed][i][epoch_idx] for seed in seeds) / len(seeds)
            avg_accs_val.append(avg_epoch_acc_val)

            # Compute standard deviation for this epoch
            variance_acc_val = sum((ACCs_val[seed][i][epoch_idx] - avg_epoch_acc_val) ** 2 for seed in seeds) / len(seeds)
            std_epoch_acc_val = math.sqrt(variance_acc_val)
            std_accs_val.append(std_epoch_acc_val)
        avg_accs_val_len.append(avg_accs_val)
        std_accs_val_len.append(std_accs_val)



    list_of_list_name = [f'loss_val_len{value}' for value in losses_val[seed]]
    # Append losses_train to the list of lists
    avg_losses_val_len.append(avg_losses_train)
    list_of_list_name.append('loss_train')
    std_losses_val_len.append(std_accs_train)

    plot_curves_2(
        avg_losses_val_len,std_losses_val_len,
        os.path.join(save_path, "Losses.pdf"),
        "All_Losses",
        legend_labels= list_of_list_name,
    )
    plot_curves_2(
        [
            avg_accs_train ,
        ],[std_accs_train],
        os.path.join(save_path, "ACCs_train.pdf"),
        "ACC Train",
        legend_labels=["ACC Train"],
    )

    list_of_list_name = [f'loss_val_len{value}' for value in losses_val[seed]]
    plot_curves_2(
        avg_accs_val_len,std_accs_val_len, os.path.join(save_path, "ACCs_val.pdf"),
        "ACC val",
        legend_labels=list_of_list_name,
    )
    print()

    #TODO: They all have different evaluation ( netwokr ) do we want ot eval ( for the average it should be ifne)
    #TODO: Think about nice visualisaiton
    #TODO: update the plotting for the other curves
    # I need to check the logging of the results
    # TODO. : plan a set of experiements to run, sudy how different initialisiton
    #TODO: Get a different set of valisaion lenght for each run

    # TODO : the set of seed changes every run. so it is fine. The question is
    #TODO: What is the the best way to have the dedges no features