
import argparse
import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from neuralplayground.agents.domine_2023_extras.class_grid_run_config import GridConfig

from neuralplayground.agents import Domine2023
from neuralplayground.agents.domine_2023_extras.class_utils import (
    rng_sequence_from_rng,
    set_device,
)

# @title Graph net functions
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path",
    metavar="-C",
    default="domine_2023_extras/Different_seed_configs/class_config_1.yaml",
    help="path to base configuration file.",
)

args = parser.parse_args()
set_device()
config_class = GridConfig
config = config_class(args.config_path)

# Init environment
arena_x_limits = [-100, 100]
arena_y_limits = [-100, 100]


agent_1 = Domine2023(
    experiment_name=config.experiment_name,
    train_on_shortest_path=config.train_on_shortest_path,
    resample=config.resample,  # @param
    wandb_on=config.wandb_on,
    seed=config.seed,
    feature_position=config.feature_position,
    weighted=config.weighted,
    num_hidden=config.num_hidden,  # @param
    num_layers=config.num_layers,  # @param
    num_message_passing_steps=config.num_message_passing_steps,  # @param
    learning_rate=config.learning_rate,  # @param
    num_training_steps=config.num_training_steps,  # @param
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
)

for n in range(config.num_training_steps):
    agent_1.update()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path",
    metavar="-C",
    default="domine_2023_extras/Different_seed_configs/class_config_2.yaml",
    help="path to base configuration file.",
)
args = parser.parse_args()
config = config_class(args.config_path)
agent_2 = Domine2023(
    experiment_name=config.experiment_name,
    train_on_shortest_path=config.train_on_shortest_path,
    resample=config.resample,  # @param
    wandb_on=config.wandb_on,
    seed=config.seed,
    feature_position=config.feature_position,
    weighted=config.weighted,
    num_hidden=config.num_hidden,  # @param
    num_layers=config.num_layers,  # @param
    num_message_passing_steps=config.num_message_passing_steps,  # @param
    learning_rate=config.learning_rate,  # @param
    num_training_steps=config.num_training_steps,  # @param
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
)

for n in range(config.num_training_steps):
    agent_2.update()



import statistics
print(statistics.mean([agent_1.roc_aucs_train,agent_2.roc_aucs_train]))
print(statistics.stdev([agent_1.roc_aucs_train,agent_2.roc_aucs_train]))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path",
    metavar="-C",
    default="domine_2023_extras/Different_seed_configs/class_config_3.yaml",
    help="path to base configuration file.",
)
args = parser.parse_args()
config = config_class(args.config_path)
agent_3 = Domine2023(
    experiment_name=config.experiment_name,
    train_on_shortest_path=config.train_on_shortest_path,
    resample=config.resample,  # @param
    wandb_on=config.wandb_on,
    seed=config.seed,
    feature_position=config.feature_position,
    weighted=config.weighted,
    num_hidden=config.num_hidden,  # @param
    num_layers=config.num_layers,  # @param
    num_message_passing_steps=config.num_message_passing_steps,  # @param
    learning_rate=config.learning_rate,  # @param
    num_training_steps=config.num_training_steps,  # @param
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
)

for n in range(config.num_training_steps):
    agent_3.update()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path",
    metavar="-C",
    default="domine_2023_extras/Different_seed_configs/class_config_4.yaml",
    help="path to base configuration file.",
)
args = parser.parse_args()
config = config_class(args.config_path)
agent_4 = Domine2023(
    experiment_name=config.experiment_name,
    train_on_shortest_path=config.train_on_shortest_path,
    resample=config.resample,  # @param
    wandb_on=config.wandb_on,
    seed=config.seed,
    feature_position=config.feature_position,
    weighted=config.weighted,
    num_hidden=config.num_hidden,  # @param
    num_layers=config.num_layers,  # @param
    num_message_passing_steps=config.num_message_passing_steps,  # @param
    learning_rate=config.learning_rate,  # @param
    num_training_steps=config.num_training_steps,  # @param
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
)

for n in range(config.num_training_steps):
    agent_4.update()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path",
    metavar="-C",
    default="domine_2023_extras/Different_seed_configs/class_config_5.yaml",
    help="path to base configuration file.",
)
args = parser.parse_args()
config = config_class(args.config_path)
agent_5 = Domine2023(
    experiment_name=config.experiment_name,
    train_on_shortest_path=config.train_on_shortest_path,
    resample=config.resample,  # @param
    wandb_on=config.wandb_on,
    seed=config.seed,
    feature_position=config.feature_position,
    weighted=config.weighted,
    num_hidden=config.num_hidden,  # @param
    num_layers=config.num_layers,  # @param
    num_message_passing_steps=config.num_message_passing_steps,  # @param
    learning_rate=config.learning_rate,  # @param
    num_training_steps=config.num_training_steps,  # @param
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
)

for n in range(config.num_training_steps):
    agent_5.update()



import statistics
print(statistics.mean([agent_1.roc_aucs_train,agent_2.roc_aucs_train,agent_3.roc_aucs_train,agent_4.roc_aucs_train,agent_5.roc_aucs_train]))
print(statistics.stdev([agent_1.roc_aucs_train,agent_2.roc_aucs_train,agent_3.roc_aucs_train,agent_4.roc_aucs_train,agent_5.roc_aucs_train]))