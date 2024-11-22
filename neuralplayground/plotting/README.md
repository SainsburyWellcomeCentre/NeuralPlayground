# Plotting

* [1 Introduction](#1-Introduction)
* [2 Model Implemented](#2-Model-Implemented)
* [3 How to Contribute](#3-How-to-Contribute)


## 1. Introduction

This class provides functionality for displaying the results of experiments and summarizing metrics or collected data. 


## 2. Plotting Functions Implemented

  1. [make_plot_trajectories](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/blob/style-guide/neuralplayground/plotting/plot_utils.py)
      * Plots the position of an agent over time. 
  2. [make_plot_rate_map](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/blob/style-guide/neuralplayground/plotting/plot_utils.py)
      * Plots the number of spikes falling within each time bin during a recording session. 
  3. [render_pl_table](https://stackoverflow.com/questions/19726663/how-to-save-the-pandas-dataframe-series-data-as-a-figure)
      * Renders a Pandas Dataframe table as an image.
  4. [make_agent_comparison](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/blob/style-guide/neuralplayground/plotting/plot_utils.py)
      * Compares agents trajectories and firing rates against experimental data.

## 3. How-to-Contribute

  1. Add your desired plotting utility to "plot_utils.py" as a function.

  2. Document what you have implemented above by describing at a high level the purpose of your plotting function.

  3. Record your contribution

All contributions should be submitted through a pull request that we will later access.
Before sending a pull request, make sure you have the following:

1. Checked the Licensing frameworks.

2. Followed the [Style Guide](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/blob/main/documents/style_guide.md).

3. Implemented and ran [Test](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/tree/main/tests).

4. Comment your work

All contributions to the repository are acknowledged through the all-contributors bot.
