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
from neuralplayground.agents.domine_2023_extras_2.class_grid_run_config import GridConfig
from neuralplayground.agents.domine_2023_extras_2.utils.utils import set_device
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

mps = [0,1,2,3,4,5,6,7]
losses_train = {mp: [] for mp  in mps}
losses_val = {mp: [] for mp  in mps}
ACCs_train = {mp: [] for mp  in mps}
ACCs_val = {mp: [] for mp in mps}

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

for mp in mps:
    agent = Domine2023(
        experiment_name=config.experiment_name,
        wandb_on=config.wandb_on,
        seed=config.seed,
        dataset=config.dataset,
        num_hidden=config.num_hidden,
        num_layers=config.num_layers,
        num_message_passing_steps= mp,
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
    losses_train[mp] = losse_train
    losses_val[mp] = losse_val
    ACCs_train[mp]= ACC_train
    ACCs_val[mp] = ACC_val

for i in config.num_nodes_max_test:
    third_elements = {key: value[i] for key, value in losses_val.items()}
    list_of_lists = list( third_elements.values())
    file_name = f"Losses_val_len{i}.pdf"
    list_of_list_name = [ f"mp_{i}.pdf" for i in list(third_elements.keys())]

    plot_curves(
        list_of_lists,
        os.path.join(save_path, file_name),
        f"Losses_val{i}",
        legend_labels=list_of_list_name,
    )

list_of_lists = list(losses_train.values())
file_name = f"Losses_train_mp.pdf"
list_of_list_name = [f"mp_{i}.pdf" for i in list(losses_train.keys())]
plot_curves(
        list_of_lists,
        os.path.join(save_path, file_name),
        "Losses_train_mp",
        legend_labels=list_of_list_name,
    )

list_of_lists = list(ACCs_train.values())
file_name = f"ACCs_mp.pdf"
list_of_list_name = [f"mp_{i}.pdf" for i in list(ACCs_train.keys())]
plot_curves(
        list_of_lists,
        os.path.join(save_path, file_name),
        "ACC_mp",
        legend_labels=list_of_list_name,
    )


