import submitit
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from neuralplayground.agents.domine_2023_extras.class_grid_run_config import GridConfig

from neuralplayground.agents import Domine2023
from neuralplayground.agents.domine_2023_extras.class_utils import (
    set_device,
)


def submit_it_function(path):
    set_device()
    config_class = GridConfig
    path = os.getcwd() + path
    print(path)
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
        grid=config.grid,
        plot=config.plot,
        dist_cutoff=config.dist_cutoff,
        n_std_dist_cutoff=config.n_std_dist_cutoff,
    )

    for n in range(config.num_training_steps):
        agent.update()


# executor is the submission interface (logs are dumped in the folder)
log_folder = "log_test/%j"
executor = submitit.AutoExecutor(folder="log_test")
# the following line tells the scheduler to only run\
# at most 2 jobs at once. By default, this is several hundreds

# set timeout in min, and partition for running the job
executor.update_parameters(timeout_min=120,mem_gb=2)
path = [
    "/domine_2023_extras/class_config.yaml",
    "/domine_2023_extras/class_config_1.yaml",
    "/domine_2023_extras/class_config_2.yaml",
    "/domine_2023_extras/class_config_3.yaml",
    "/domine_2023_extras/class_config_4.yaml",
    "/domine_2023_extras/class_config_5.yaml",
    "/domine_2023_extras/class_config_6.yaml",
    "/domine_2023_extras/class_config_7.yaml",
    "/domine_2023_extras/class_config_8.yaml",
    "/domine_2023_extras/class_config_9.yaml",
    "/domine_2023_extras/class_config_10.yaml",
    "/domine_2023_extras/class_config_11.yaml",
    "/domine_2023_extras/class_config_12.yaml",
    "/domine_2023_extras/class_config_13.yaml",
    "/domine_2023_extras/class_config_14.yaml",
    "/domine_2023_extras/class_config_15.yaml",
    "/domine_2023_extras/class_config_16.yaml",
    "/domine_2023_extras/class_config_17.yaml",
    "/domine_2023_extras/class_config_18.yaml",
    "/domine_2023_extras/class_config_19.yaml",
    "/domine_2023_extras/class_config_20.yaml",
    "/domine_2023_extras/class_config_21.yaml",
    "/domine_2023_extras/class_config_22.yaml",
    "/domine_2023_extras/class_config_23.yaml",
    "/domine_2023_extras/class_config_24.yaml",
]

job = executor.map_array(submit_it_function, path)  # will compute add(5, 7)
print(job[0].job_id)  # ID of your job
print(job[1].job_id)



