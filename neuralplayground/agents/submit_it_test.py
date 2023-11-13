# TODO: NOTE to self: This is a work in progress, it has not been tested to work, I think Jax is not a good way to implement in object oriented coding.
# I think if I want to implement it here I should use neuralplayground it would be in pytorch.

import argparse
import submitit
import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from neuralplayground.agents.domine_2023_extras.class_grid_run_config import GridConfig

from neuralplayground.agents import Domine2023
from neuralplayground.agents.domine_2023_extras.class_utils import (
    rng_sequence_from_rng,
    set_device,
)

def submit_it_function(path):

    set_device()
    config_class = GridConfig
    config = config_class(path)

    # Init environment
    arena_x_limits = [-100, 100]
    arena_y_limits = [-100, 100]

    agent = Domine2023(
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
        agent.update()


# executor is the submission interface (logs are dumped in the folder)
log_folder = "log_test/%j"
executor = submitit.AutoExecutor(folder="log_test")
# the following line tells the scheduler to only run\
# at most 2 jobs at once. By default, this is several hundreds

# set timeout in min, and partition for running the job
executor.update_parameters(timeout_min=50, slurm_partition="dev")
path =["/Users/clementine/Documents/UCL/NeuralPlayground/neuralplayground/agents/domine_2023_extras/class_config.yaml","/Users/clementine/Documents/UCL/NeuralPlayground/neuralplayground/agents/domine_2023_extras/class_config.yaml"]
job = executor.map_array(submit_it_function,path )  # will compute add(5, 7)
print(job[0].job_id)  # ID of your job
print(job[1].job_id)
output = job[0].result()  # waits for completion and returns output
 # 5 + 7 = 12...  your addition was computed in the cluster