# Agent

* [1 Introduction](#1-Introduction)
* [2 Models Implemented](#2-Model-Implemented)
* [3 How to Contribute](#3-How-to-Contribute)


## 1. Introduction

This class includes a set of functions that control the ways intelligent systems interact
with their surroundings (i.e., the environment). An agent receives observations from the environment (reward, visual cues, etc.) and uses these to take an action which in turn will update both its state and the state of the environment, generating new observations. More generally, the Agent can be thought of as an animal performing the task in the simulated experiment. All agent types will be given a set of abilities that are summarised in the agent’s main class. Each different model developed can be easily implemented as a subclass of the main one.


## 2. Models-Implemented

  1. [The hippocampus as a predictive map](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/blob/main/neuralplayground/agents/stachenfeld_2018.py)

  2. [Learning place cells, grid cells and invariances with excitatory and inhibitory plasticity](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/blob/main/neuralplayground/agents/weber_2018.py)

  3. [The Tolman-Eichenbaum Machine: Unifying Space and Relational Memory through Generalization in the Hippocampal Formation](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/blob/main/neuralplayground/agents/whittington_2020.py)


Each of the these models are implemented in a jupyter notebook in [Examples](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/tree/main/examples) to facilitate the interaction.

Pre-trained version of some of these model are hosted on a separate data repository on GIN. GIN offers an interface almost identical to GitHub. 
To contribute a new trainned model, you need to fork the repository and open a pull request, just like on GitHub.
Place your model trained in a selected arena folder named as "author_date_in_arena", zip the folder, and place "author_date_in_arena.zip" under the "data" directory of the Forked repository, for example, "data/smith_2023_in_Simple2D.zip". 
If you encounter any problems with this procedure, do not hesitate to contact us.

## 3. How-to-Contribute

  1. Create a file that indicates the appropriate reference to the model

  2. Create a class to run the model following the template shown in the [Agent_core.py](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/blob/main/neuralplayground/agents/agent_core.py) and the [Style Guide](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/blob/main/documents/style_guide.md).
  When building a new model create a file named author_date.py.

  3. Create or add to [Examples](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/tree/main/examples/agent_examples) jupyter notebook for the new model where you can run the model in a chosen environment with selected experimental data

  4. Record your contribution

All contributions should be submitted through a pull request that we will later access.
Before sending a pull request, make sure you have the following:

1. Checked the Licensing frameworks.

2. Followed the [Style Guide](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/blob/main/documents/style_guide.md).

3. Implemented and ran [Test](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/tree/main/tests).

4. Comment your work

All contributions to the repository are acknowledged through the all-contributors bot.
