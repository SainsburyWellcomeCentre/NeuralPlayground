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
from neuralplayground.agents.domine_2023_2 import Domine2023

# from neuralplayground.agents.domine_2023_extras_2.evaluate import Evaluator
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


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
        batch_size_test=config.batch_size_test,
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
# avg_losses_val_len.append(avg_losses_train)
# list_of_list_name.append('loss_train')
# std_losses_val_len.append(std_accs_train)

plot_curves_2(
    avg_losses_val_len,std_losses_val_len,
    os.path.join(save_path, "Losses.pdf"),
    "All_Losses",
    legend_labels = list_of_list_name,
)
# Append losses_train to the list of lists
# avg_losses_val_len.append(avg_losses_train)
# list_of_list_name.append('loss_train')
# std_losses_val_len.append(std_accs_train)

plot_curves_2(
    [avg_losses_train], [std_losses_train],
    os.path.join(save_path, "Losses_train.pdf"),
    "All_Losses",
    legend_labels = "Loss Train",
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
#TODO: What is the the best way to have the dedges no features    if plot:
#os.makedirs(os.path.join(self.save_path, "results"), exist_ok=True)
#self.save_path = os.path.join(self.save_path, "results")
#    file_name = f"Losses_{seed}.pdf"

    # Combine the path and file name
#  list_of_lists = [value for value in self.losses_val.values()]
#  list_of_list_name = [f'loss_val_len{value}' for value in self.losses_val]
#  # Append losses_train to the list of lists
#  list_of_lists.append(self.losses_train)
#  list_of_list_name.append('loss_train')

#  plot_curves(
#      list_of_lists,
#      os.path.join(self.save_path, file_name),
#    "All_Losses",
#    legend_labels=list_of_list_name,
#  )

#  file_name = f"ACCs_val_{seed}.pdf"
#  plot_curves([value for value in self.ACCs_val.values()],
#      os.path.join(self.save_path, file_name),
#      "ACC Val",
#      legend_labels=[f'ACC_val_len{value}' for value in self.losses_val],
#      )
#   file_name = f"ACCs_train_{seed}.pdf"

#  pl#ot_curves(
#  [
            #       #        self.ACCs_train,
#  ],
#  os.path.join(self.save_path, file_name),
#    "ACC train",
#    legend_labels=["ACC train"],
# )
