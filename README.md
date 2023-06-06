# NeuralPlayground: The  standardised environment for the hippocampus and entorhinal cortex models.
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-6-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->



* [1 Introduction](#1-Introduction)
* [2 Installation ](#2-Installation)
* [3 The Project](#3-Project)
* [4 How to Contribute](#4-I-want-to-Contribute)
* [5 Cite ](#5-Cite)
* [6 Licence](#6-License)

## 1. Introduction
The abstract representation of space has been extensively studied in the hippocampus and entorhinal cortex in part due to the easy monitoring of the task and neural recording. A growing variety of theoretical models have been proposed to capture the rich neural and behavioral phenomena
associated with these circuits. However, objective comparison of these theories against each other and against empirical data is difficult.

Although the significance of virtuous interaction between experiments and theory is widely recognized, the tools available to facilitate comparison are limited. Some important challenge to standardized coparaison are the

   1. Lack availability and accessibility of data in a standardized, labeled format.

   2. Lack of standard or easy ways for models to interact with the task.

   3. Lack  of standard or easy ways to compare model output with empirical data.

To address this gap, we present an open-source standardised software framework - NeuralPlayground - to enable adjudication between the hippocampus
and entorhinal cortex models. This Python software package offers a reproducible way to compare models against a centralised library of published experimental results, including neural recordings and animal behavior.
The framework currently contains implementations of three Agents, as well as three Experiments providing simple interfaces to publicly available neural and behavioral data. It also contains a customizable 2-dimensional Arena (continuous and discrete) able to produce common experimental environments in which the agents can move in and interact with. We note that each module can also be used separately, allowing flexible access to influential models and data sets.

We currently rely on visual comparison of a hand-selected number of outputs of the model with neural recordings as shown in [github.com/NeuralPlayground/examples/comparison](https://github.com/ClementineDomine/NeuralPlayground/blob/main/examples/comparison_board_examples/comparison_board.ipynb). In the future, a set of quantitative measures and qualitative measures will be added for systematic comparisonsk from any Agent, Arena, Experiments.We want to restate that this won’t constitute an objective judgment of the quality of an Agent to replicate the brain mechanism. Instead, this only allows an objective and complete comparison to the current evidence in the field, as is typically done in publications.

Altogether, we hope our framework, available at [github.com/NeuralPlayground](https://github.com/ClementineDomine/NeuralPlayground/), offers
a foundation that the community will build upon, working toward a shared, standardized, open, and
reproducible computational understanding of the hippocampus and entorhinal cortex.

## 2. Installation
You can create a new environment using conda, and the yml file with all the right
dependencies by running
```
git clone https://github.com/SainsburyWellcomeCentre/NeuralPlayground/
conda create --name NPG
conda activate NPG
conda install pip
```

For now, install using pip for local editing and testing
```
pip install -e .
```
or
```
pip install -e '.[dev]'
```
if you want to install the dependencies for development.

## 3. Project

Try our package! We are gathering opinions to focus our efforts on improving aspects of the code or adding new features, so if you tell us what you would like to have, we might just implement it ;) Please refer to the [Roadmap](https://github.com/ClementineDomine/NeuralPlayground/blob/main/documents/road_map.md) to understand the state of the project and get an idea of the direction it is going in. This open-source software was built to be collaborative and lasting. We hope that the framework will be simple enough to be adopted by a great number of neuroscientists, eventually guiding the path to the computational understanding of the HEC mechanisms. We follow reproducible, inclusive, and collaborative project design guidelines. All relevant documents can be found in [Documents](https://github.com/ClementineDomine/NeuralPlayground/blob/main/documents/).

#### How to run a single module

Each module can be used separately to easily explore and analyze experimental data and better understand any implemented model. Additionally, different Arenas can be initialised with artificial architectures or with data from real-life experiments. We provide examples of module instantiation in the detailed jupyter notebooks found in [Examples_experiment](https://github.com/ClementineDomine/NeuralPlayground/tree/main/examples/experimental_examples), [Examples_arena](https://github.com/ClementineDomine/NeuralPlayground/tree/main/examples/arena_examples) and [Examples_agents](https://github.com/ClementineDomine/NeuralPlayground/tree/main/examples/agent_examples).

#### How to run interactions between modules

As shown in the jupyter notebooks [Examples_agent](https://github.com/ClementineDomine/NeuralPlayground/tree/main/examples/agent_examples), the Agent can interact with an Arena in a standard RL framework. The first step is to initialise an Agent and Arena of your choice. The Agent can be thought of as the animal performing the Experiment and the Arena as the experimental setting where the animal navigates and performs a task.

#### How to run comparisons

As shown in the jupyter notebooks [Examples_comparison](https://github.com/ClementineDomine/NeuralPlayground/blob/main/examples/comparison_board_examples/comparison_board.ipynb). We show visual comparisons between results from agents running with experimental behavior and results from the real experiment.

### Check our Tolman-Eichenbaum Machine Implementation in [this branch](https://github.com/ClementineDomine/NeuralPlayground/tree/whittington_2020) (work in progress).

## 4. I-want-to-Contribute

There are many ways to contribute to the [github.com/NeuralPlayground/examples/comparison](https://github.com/ClementineDomine/NeuralPlayground/tree/main/neuralplayground).

 1. Implement a hippocampal and entorhinal cortex [Agent](https://github.com/ClementineDomine/NeuralPlayground/tree/main/neuralplayground/agents) of your choice.

 2. Work on improving the [Arena](https://github.com/ClementineDomine/NeuralPlayground/tree/main/neuralplayground/arenas).

 3. Add an [Experimental](https://github.com/ClementineDomine/NeuralPlayground/tree/main/neuralplayground/experiments) data set.

All contributions should be submitted through a pull request that we will later access.
Before sending a pull request, make sure you have the following:

1. Checked the Licensing frameworks.

2. Followed the [PEP8](https://www.python.org/dev/peps/pep-0008/) and [numpy docstring](https://numpydoc.readthedocs.io/en/latest/format.html) style convention. More details found in [Style Guide](https://github.com/ClementineDomine/NeuralPlayground/tree/main/documents/style_guide.md).

3. Implemented and ran [Test](https://github.com/ClementineDomine/NeuralPlayground/tree/main/neuralplayground/tests).

4. Comment your work.

All contributions to the repository are acknowledged through the all-contributors bot.
Refer to the README.md files found in each of the modules for further details on how to contribute to them.


## 5. Cite

See [Citation](https://github.com/ClementineDomine/NeuralPlayground/blob/main/documents/citation.cff) for the correct citation of this framework.

## 6. License

More details about the license can be found at [Licence](https://github.com/ClementineDomine/NeuralPlayground/blob/main/documents/lisence.md).


## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ClementineDomine"><img src="https://avatars.githubusercontent.com/u/18595111?v=4?s=100" width="100px;" alt="Clementine Domine"/><br /><sub><b>Clementine Domine</b></sub></a><br /><a href="#design-ClementineDomine" title="Design">🎨</a> <a href="#mentoring-ClementineDomine" title="Mentoring">🧑‍🏫</a> <a href="https://github.com/SainsburyWellcomeCentre/NeuralPlayground/commits?author=ClementineDomine" title="Code">💻</a> <a href="#data-ClementineDomine" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/rodrigcd"><img src="https://avatars.githubusercontent.com/u/22643681?v=4?s=100" width="100px;" alt="rodrigcd"/><br /><sub><b>rodrigcd</b></sub></a><br /><a href="#design-rodrigcd" title="Design">🎨</a> <a href="#mentoring-rodrigcd" title="Mentoring">🧑‍🏫</a> <a href="https://github.com/SainsburyWellcomeCentre/NeuralPlayground/commits?author=rodrigcd" title="Code">💻</a> <a href="#data-rodrigcd" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/LukeHollingsworth"><img src="https://avatars.githubusercontent.com/u/93782020?v=4?s=100" width="100px;" alt="Luke Hollingsworth"/><br /><sub><b>Luke Hollingsworth</b></sub></a><br /><a href="https://github.com/SainsburyWellcomeCentre/NeuralPlayground/commits?author=LukeHollingsworth" title="Documentation">📖</a> <a href="https://github.com/SainsburyWellcomeCentre/NeuralPlayground/commits?author=LukeHollingsworth" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://saxelab.org"><img src="https://avatars.githubusercontent.com/u/4165949?v=4?s=100" width="100px;" alt="Andrew Saxe"/><br /><sub><b>Andrew Saxe</b></sub></a><br /><a href="#mentoring-asaxe" title="Mentoring">🧑‍🏫</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/DrCaswellBarry"><img src="https://avatars.githubusercontent.com/u/17472149?v=4?s=100" width="100px;" alt="DrCaswellBarry"/><br /><sub><b>DrCaswellBarry</b></sub></a><br /><a href="#mentoring-DrCaswellBarry" title="Mentoring">🧑‍🏫</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://nikosirmpilatze.com"><img src="https://avatars.githubusercontent.com/u/20923448?v=4?s=100" width="100px;" alt="Niko Sirmpilatze"/><br /><sub><b>Niko Sirmpilatze</b></sub></a><br /><a href="#infra-niksirbi" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-niksirbi" title="Maintenance">🚧</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind are welcome!
