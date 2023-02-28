# Arena

* [1 Introduction](#1-Introduction)
* [2 Arena Implemented](#2-Arena-Implemented)
* [3 How to Contribute](#3-How-to-Contribute)

## 1. Introduction

Arena provides an environment in which an agent can explore and potentially learn
over time, reproducing aspects of the physical layout of an experimental paradigm in which behavioral and neural data were collected. Any two-dimensional discrete and continuous Arenas can be built
using walls as construction units. This allows complex experimental architectures such as connected rooms, T-mazes
or cycles to be added. Dynamical arenas, such as the merging room experiment in Wernle et
al. (2018) can also be implemented. Each specific environment implemented to resemble an experimental setting should be created as a subclass of the main environment class. The Environment can be initialised with data from real-life experiments. We will work toward improving each of
the Environments through the projects, adding experimental specifications.

## 2. Arena-Implemented

1. [Any 2D Environmement](https://github.com/ClementineDomine/NeuralPlayground/blob/main/neuralplayground/arenas/simple2d.py)
2. [Merging Room](https://github.com/ClementineDomine/NeuralPlayground/blob/main/neuralplayground/arenas/connected_rooms.py)

## 3. How-to-Contribute

1. Create an environment class following the template shown in the [Arena_Core](https://github.com/ClementineDomine/NeuralPlayground/blob/main/neuralplayground/arenas/arena_core.py) and the [Style Guide](https://github.com/ClementineDomine/NeuralPlayground/tree/main/documents/style_guide.md). 
Any two dimentional arena should inherit from the 2D simple [Simple2d](https://github.com/ClementineDomine/NeuralPlayground/blob/main/neuralplayground/arenas/simple2d.py) which has inbuild functions for 2dimentional environments.
To build 2D experimental arena, create a file named anthor_date.py as a child class of 2D simple.

2. Create or add to [Examples](https://github.com/ClementineDomine/NeuralPlayground/tree/main/examples/arena_examples//) jupyter notebook for the new arena.

3. Record your contribution.

All contributions should be submitted through a pull request that we will later access. 
Before sending a pull request, make sure you have the following: 

1. Checked the Licensing frameworks. 

2. Followed the [Style Guide](https://github.com/ClementineDomine/NeuralPlayground/tree/main/documents/style_guide.md).

3. Implemented and ran [Test](https://github.com/ClementineDomine/NeuralPlayground/tree/main/neuralplayground/tests).

4. Commented your work 
    
All contributions to the repository are acknowledged through the all-contributors bot and in a future publication.
