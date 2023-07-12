# Experiment

- [Experiment](#experiment)
  - [1. Introduction](#1-introduction)
  - [2. Experiment-Implemented](#2-experiment-implemented)
  - [3. How-to-Contribute](#3-how-to-contribute)

## 1. Introduction

This class gives access to open-source experimental data (neural recording, behaviours, etc.) through various plotting
functions and visualisations of experimental measurements. Each data set is organised in recordings session with an
attributed recording number (rec index), given in a list at the initialisation of the class. It is possible to plot a
selected tetrode (select the index in the list) recording ```plot_recording_tetr()```, the trajectory recording
within the arena ```plot_trajectory()``` and get access to the experimental details call ```show_keys()```.
For further explanation of the datasets, check the [notebook examples](https://github.com/ClementineDomine/NeuralPlayground/blob/main/examples/experimental_examples/experimental_data_examples.ipynb) using this class.

## 2. Experiment-Implemented

We use the following data sets:
 > [”Conjunctive Representation of Position, Direction, and Velocity in Entorhinal Cortex” Sargolini et al.
 > 2006.](https://github.com/ClementineDomine/NeuralPlayground/blob/main/neuralplayground/experiments/sargolini_2006_data.py)

 > [”Hippocampus-independent phase precession in entorhinal grid cells”, Hafting et al
 > 2008](https://github.com/ClementineDomine/NeuralPlayground/blob/main/neuralplayground/experiments/hafting_2008_data.py)

 > [”Integration of grid maps in merged environments”, Wernle et al,
 > 2018](https://github.com/ClementineDomine/NeuralPlayground/blob/main/neuralplayground/experiments/wernle_2018_data.py)

One of our goals is to expand this list to add more experiments that are relevant to this literature and that are publicly available.

## 3. How-to-Contribute

1. The experimental data are hosted on a [separate data repository on GIN](https://gin.g-node.org/SainsburyWellcomeCentre/NeuralPlayground). GIN offers an interface almost identical to GitHub. To contribute a new dataset, you need to fork the repository and open a pull request, just like on GitHub. Place yout data in a folder named as "author_date", zip the folder, and place "author_date.zip" under the "data" directory of the Forked repository, for example, "data/smith_2023.zip". If you encounter any problems with this procedure, do not hesitate to contact us.

2. Next, you need to update the `DATASET_REGISTRY` in the [`datasets.py` module](../datasets.py). You will need to add the dataset name (e.f. "smith_2023") and the corresponding sha256 hash of the zip file. You can compute the hash using the following command in the terminal: `sha256sum author_date.zip`. The hash is the first string in the output.

3. Create a class to read/filter with name author_date_data the data inheriting from the [Experiment class](https://github.com/ClementineDomine/NeuralPlayground/blob/main/neuralplayground/experiments/experiment_core.py),
which is just an abstract class to share same parent. For a 2D environment, the new data class could inherit from the
base [Hafting2008Data](https://github.com/ClementineDomine/NeuralPlayground/blob/main/neuralplayground/experiments/hafting_2008_data.py)
directly (as [Sargolini2006Data](https://github.com/ClementineDomine/NeuralPlayground/blob/main/neuralplayground/experiments/sargolini_2006_data.py) does),
which has implemented some basic functions providing that they share a similar data structure.

1. Create or add to [Examples](https://github.com/ClementineDomine/NeuralPlayground/tree/main/examples/experimental_examples/) jupyter notebook for the new experiment.

2. Add unit tests in the [test module](https://github.com/ClementineDomine/NeuralPlayground/tree/main/neuralplayground/tests)

3. Cite the data appropriately. Your contribution will be automatically considered by our bot once the pull request has been accepted to the main branch.


All contributions should be submitted through a pull request that we will later access.
Before sending a pull request, make sure you have the following:
1. Checked the Licensing frameworks.

2. Followed the [Style Guide](https://github.com/ClementineDomine/NeuralPlayground/tree/main/documents/style_guide.md).

3. Implemented and ran [Test](https://github.com/ClementineDomine/NeuralPlayground/tree/main/neuralplayground/tests).

4. Commented your work

All contributions to the repository are acknowledged through the all-contributors bot.
