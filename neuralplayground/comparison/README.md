# Comparison

* [1 Introduction](#1-Introduction)
* [2 Metrics Implemented](#2-Model-Implemented)
* [3 How to Contribute](#3-How-to-Contribute)


## 1. Introduction

This class includes a set of functions that will help when comparing various models against experimental data. More specifically these metrics will compare an agent's neural representations of the environment against other agents or experimental data. For the moment GridScore is the only metric implemented through the `GridScorer` class.


## 2. Metrics-Implemented

  1. [GridScore](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/blob/style-guide/neuralplayground/comparison/metrics.py)


For an example of the use of the `GridScorer` class see [the Jupyter Notebook](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/blob/main/examples/comparisons_examples/comparison_examples_score.ipynb).

## 3. How-to-Contribute

  1. Create a file that indicates the appropriate reference to the metric.

  2. Create a class which implements the metric given neural representations.

  3. Create or add to [Examples](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/tree/main/examples/agent_examples) jupyter notebook for the new metric where you can compare models in a chosen environment with selected experimental data.

  4. Record your contribution

All contributions should be submitted through a pull request that we will later access.
Before sending a pull request, make sure you have the following:

1. Checked the Licensing frameworks.

2. Followed the [Style Guide](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/blob/main/documents/style_guide.md).

3. Implemented and ran [Test](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/tree/main/tests).

4. Comment your work

All contributions to the repository are acknowledged through the all-contributors bot.
