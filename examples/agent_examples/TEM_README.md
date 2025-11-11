# Tolman-Eichenbaum Machine (TEM) Model

The Tolman-Eichenbaum Machine (TEM) is a computational model of hippocampal and entorhinal function, inspired by the cognitive maps proposed by Edward Tolman and the grid cells discovered by John O'Keefe and Lynn Nadel. This model aims to understand the neural mechanisms underlying spatial navigation and memory and is an implementation of the work done by J. Whittington, T. Muller, S. Mark, G. Chen, C. Barry, N. Burgess and T. Behrens in their paper [Tolman-Eichenbaum Machine: A Model of Hippocampal and Entorhinal Function](https://www.sciencedirect.com/science/article/pii/S009286742031388X).


## Running the Model

To run the TEM model, follow the steps below:

1. Clone the repository and install the correct dependencies to a virtual environment, as shown in the main [README](../../README.md).
2. Install the additional packages, required by TEM, as follows:
    ```
    conda activate NPG-env
    pip install torch tensorboard
    ```
3. Open the running file ([running.py](../agent_examples/whittington_2020_run.py)) to set up basic model parameters and environment specifications. All TEM model parameters can be found and modified in the [whittington_2020_parameters.py](../../neuralplayground/agents/whittington_2020_extras/whittington_2020_parameters.py) file. Modifying both of these is discussed further below.
4. Modify the parameters as desired, such as changing the shape or size of the environment and enabling the use of behavioral data.
5. Run the model by executing the running file.

**Important Notes:**

- In order to generate the correct path for the saving of a trained TEM model, be sure to run the [whittington_2020_run.py](../agent_examples/whittington_2020_run.py) file from its location at `NeuralPlayground/examples/agent_examples/`.
- Running the full TEM model may require significant computational resources and time.
- Pretrained models are provided for convenience, allowing you to explore the results without training the model from scratch.
Pre-trained version of the model are hosted on a separate data repository on GIN. GIN offers an interface almost identical to GitHub.
To contribute a new trainned model, you need to fork the repository and open a pull request, just like on GitHub.
Place your model trained in a selected arena folder named as "author_date_in_arena", zip the folder,
and place "author_date_in_arena.zip" under the "data" directory of the Forked repository, for example,"data/smith_2023_in_Simple2D.zip".
If you encounter any problems with this procedure, do not hesitate to contact us.

### Modifying Parameters

In the running file, you can modify the following parameters of the TEM model:

- `batch_size`: Number of environments to run in parallel.
- `arena_x_limits` and `arena_y_limits`: The x and y limits of the environments' boundaries.
- `agent_step_size`: Length factor by which each agent action is scaled by.
- `state_density`: Density of states in the environment. Higher density results in more fine-grained states.

The [parameters file](../../neuralplayground/agents/whittington_2020_extras/whittington_2020_parameters.py) gives much more control over the intricacies of the Tolman-Eichenbaum machine. Some example parameters are given below:
- `n_f_g`: Number of hierarchical frequency modules for grid cells.
- `n_x`: Number of neurons for sensory observation x. It defines the dimensionality of the sensory input space.
- `lambda` and `eta`: Hebbian rates of forgetting and remembering, respectively.
- `i_attractor`: Number of iterations of attractor dynamics for memory retrieval.
- `p_update_mask` and `p_retrieve_mask_inf`: Connectivity matricies for forming and retrieving Hebbian memories of grounded locations. These define the interactions between module frequencies in the attractor network.

## Plotting Results

The [plotting file](../agent_examples/whittington_2020_plot.py) allows you to visualize the results of a trained TEM model. You can plot various aspects, such as prediction accuracy, grid cells, and place cells.

### Modifying Parameters

In the plotting file, you can modify the following parameters:

- `arena_x_limits` and `arena_y_limits`: The x and y limits of the environments' boundaries which you run the model in.
- `include_stay_still`: Boolean to determine whether null actions are included in the plotting trajectory.
- `shiny_envs`: List of booleans that specify which environments to plot.
- `env_to_plot`: Index of the environment to plot (if `shiny_envs` is set to `False` for all environments).
- `envs_to_avg`: List of booleans that specify which environments to average when plotting.

**Note:** The plotting code is designed to work with the same environment that the model was trained on. If you want to plot using a different environment, you may need to make some adjustments.

### Pretrained Models

Pretrained TEM models are available in the repository. To use them, simply load the desired model by setting the appropriate path and index in the running file. You can then explore the model's predictions and representations.

Changes to make: I want to streamline how I save and load model files in the running and plotting of TEM results, respectively. There is a standardised way we do this now which includes saving the agent and environment classes, as well as a file containing model parameters.

### Plotting Rate Maps

You can visualize the rate maps of grid or place cells using the `agent.plot_rate_map` function in the plotting file.

### Plotting Results and Zero-Shot Inference Analysis

You can plot the results of agent comparison and zero-shot inference analysis using the provided functions in the plotting file. These plots show how well the TEM model performs compared to node and edge agents and its ability to infer new environments without prior training.

## Conclusion

The Tolman-Eichenbaum Machine is a powerful model for studying spatial navigation and memory processes. By following the instructions in this README, you can run the TEM model, explore pretrained models, and visualize its predictions and representations.

For any inquiries or issues, please contact [Luke Hollingsworth](mailto:luke.hollingsworth.21@ucl.ac.uk).

Happy exploring and modeling!
