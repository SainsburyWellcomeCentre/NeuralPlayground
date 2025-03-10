# Arena

* [1 Introduction](#1-Introduction)
* [2 Arena Implemented](#2-Arena-Implemented)
* [3 How to Contribute](#3-How-to-Contribute)

## 1. Introduction

Arena provides an environment in which an agent can explore and potentially learn
over time, reproducing aspects of the physical layout of an experimental paradigm in which behavioral and neural data
were collected. Any two-dimensional discrete and continuous Arenas can be built
using walls as construction units. This allows complex experimental architectures such as connected rooms, T-mazes
or cycles to be added. Dynamical arenas, such as the merging room experiment in [Wernle et
al. (2018)](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/blob/main/neuralplayground/arenas/wernle_2018.py)
are also be implemented. Each specific environment implemented to resemble an experimental setting should be
created as a subclass of the main environment class. The Environment can be initialised with data from real-life
experiments. We will work toward improving each of the Environments through the projects, adding experimental specifications,
richer perceptual inputs and flexibility to analyze and run simulations.

Check the [arena examples notebook](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/blob/main/examples/arena_examples/arena_examples.ipynb) for further explanation.

## 2. Arenas-Implemented
1. For now, all arenas are [2D environments](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/blob/main/neuralplayground/arenas/simple2d.py) where
agents can navigate. These arenas can have arbitrary shaped walls as explained in the [jupyter notebook](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/blob/main/examples/arena_examples/arena_examples.ipynb) with
examples.
2. Some of the arenas are based on a real experimental settings by setting the parameters of the arena
to resemble the physical dimensions of the arena used in the real experiment. These arenas also
load the experimental data automatically and have some extra methods to plot and use the experimental data from the corresponding experiments. These classes are [Hafting2008](https://github.com/ClementineDomine/NeuralPlayground/blob/main/neuralplayground/arenas/hafting_2008.py),
[Sargolini2006](https://github.com/ClementineDomine/NeuralPlayground/blob/main/neuralplayground/arenas/sargolini_2006.py)
and [Wernle2018](https://github.com/ClementineDomine/NeuralPlayground/blob/main/neuralplayground/arenas/wernle_2018.py) where the arena change throughout the experiment.
3. A [discretized_objects](../arenas/discritized_objects.py) arena is also included. This allows agents to instead run in a discretised environment and generate a sequence of discrete state transitions, as opposed to a continuous trajectory. The resulting 2D arena is split into states which each have a one-hot encoded object associated with them; observations in this environment are of the form `[state_id, object, position]`. Therefore, additional parameters are required to initialise this arena, including `n_objects` and `environment_name="DiscreteObject"`. This is the arena type that the [Tolman-Eichenbaum machine](../../examples/agent_examples/TEM_README.md) runs in.
3. Also included is the [batch_environment](../arenas/batch_environment.py) class, which offers the ablity to run multiple agents simultaneously (in a batch). Note that this will only generate different trajectories when run with an artificial agent; if used with experimental data, each element of the batch will use that same data. This requires an agent that also generates batches of actions, and so is currently only used by the [Tolman-Eichenbaum machine](../agents/whittington_2020.py) agent.

## 3. How-to-Contribute

1. Create an environment class following the template shown in the [Arena_Core](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/blob/main/neuralplayground/arenas/arena_core.py) and the [Style Guide](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/blob/main/documents/style_guide.md).

2. Any 2D arena should inherit from the 2D simple [Simple2d](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/blob/main/neuralplayground/arenas/simple2d.py), which has inbuild functions for 2dimentional environments.
To build a 2D experimental arena, create a file named anthor_date.py as a child class of 2D simple.

2. Create or add to [Examples](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/tree/main/examples/arena_examples) jupyter notebook for the new arena.

3. Cite the data appropriately. Your contribution will be automatically considered by our bot once the pull request
4. has been accepted to the main branch.

All contributions should be submitted through a pull request that we will later access.
Before sending a pull request, make sure you have the following:

1. Checked the Licensing frameworks.

2. Followed the [Style Guide](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/blob/main/documents/style_guide.md).

3. Implemented and ran [Test](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/tree/main/tests).

4. Commented your work

All contributions to the repository are acknowledged through the all-contributors bot.
