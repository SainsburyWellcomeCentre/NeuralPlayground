# Experiment

* [1. Introduction](#1-Introduction)
* [2. Experiment Implemented](#2-Experiment-Implemented)
* [3. How to Contribute](#3-How-to-Contribute)

## 1. Introduction

This class gives access to open-source experimental data (neural recording, behaviours, etc.) through various plotting functions and visualisations of experimental measurements. Each data set is organised in recordings session with an attributed recording number (rec index), given in a list at the initialisation of the class. It is possible to plot a selected tetrode (select the index in the list) recording $.plot_{}recording_{}tetr(index)$, the trajectory recording within the arena $.plot_{}trajectory(index)$ and get access to the experimental details call $.show_{}keys()$.

## 2. Experiment-Implemented

We use the following data sets:
 > [”Conjunctive Representation of Position, Direction, and Velocity in Entorhinal Cortex” Sargolini et al. 2006.](https://github.com/ClementineDomine/NeuralPlayground/blob/main/neuralplayground/experiments/sargolini_2006_data.py)

 > [”Hippocampus-independent phase precession in entorhinal grid cells”, Hafting et al 2008](https://github.com/ClementineDomine/NeuralPlayground/blob/main/neuralplayground/experiments/hafting_2008_data.py)
   
 > [”Integration of grid maps in merged environments”, Wernle et al, 2018](https://github.com/ClementineDomine/NeuralPlayground/blob/main/neuralplayground/experiments/wernle_2018_data.py)
   
One of our goals is to expand this list to add more experiments that are relevant to this literature and that are publicly available.

## 3. How-to-Contribute

1. Create a directory where to download and store the data with name author_data.

2. Create a class to read/filter with name  author_date_data the data following the template shown in the [Experiment_core](https://github.com/ClementineDomine/NeuralPlayground/blob/main/neuralplayground/experiments/experiment_core.py). The [Experiment_core](https://github.com/ClementineDomine/NeuralPlayground/blob/main/neuralplayground/experiments/experiment_core.py) is not specfidc to allow for various data format. For 2D enviroment, the new data class could inherits from the base Hafting_2008, which has implemented some basic functions providing that they share similar data structure.

3. Create or add to [Examples](https://github.com/ClementineDomine/NeuralPlayground/tree/main/examples/experimental_examples/) jupyter notebook for the new experiment.

4. Cite the data appropriately.

5. Record your contribution


All contributions should be submitted through a pull request that we will later access. 
Before sending a pull request, make sure you have the following:
1. Checked the Licensing frameworks. 

2. Followed the [Style Guide](https://github.com/ClementineDomine/NeuralPlayground/tree/main/documents/style_guide.md).

3. Implemented and ran [Test](https://github.com/ClementineDomine/NeuralPlayground/tree/main/neuralplayground/tests).

4. Commented your work 

All contributions to the repository are acknowledged through the all-contributors bot and in future publications.

