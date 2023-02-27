# The  standardised environment for the hippocampus and entorhinal cortex models. 
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-5-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->



* [1 Introduction](#1-Introduction)
* [2 Installation ](#2-Installation)
* [3 The Project](#3-Project)
* [4 How to Contribute](#4-I-want-to-Contribute)
* [5 Cite ](#5-Cite)
* [6 Lisence ](#6-Lisence)

## 1. Introduction
The abstract representation of space has been extensively studied in the hippocampus and entorhinal cortex in part due to the easy monitoring of the task and neural recording. A growing variety
of theoretical models have been proposed to capture the rich neural and behavioral phenomena
associated with these circuits. However, objective comparison of these theories against each other
and against empirical data is difficult. 

To address this gap, we present an open-source standardised
software framework - NeuralPlayground - to enable adudication between models of hippocampus
and entorhinal cortex. This Python software package offers a reproducible way to compare models
against a centralised library of published experimental results, including neural recordings and animal behavior. The framework currently contains implementations of three Agents, as well as three
Experiments providing simple interfaces to publicly available neural and behavioral data. It also
contains a customizable 2-dimensional Arena (continuous and discrete) able to produce common
experimental environments which the agents can move in and interact with and a Comparison tool
to facilitate systematic comparisons by enabling users to pick from any Agent, Arena, Experiments. Each modules can also be used separately, allowing flexible access to influential models and data sets. 

All together, we hope our framework, available at [github.com/NeuralPlayground](https://github.com/ClementineDomine/NeuralPlayground/), offers
a foundation that the community will build upon, working toward a shared standardized, open, and
reproducible computational understanding of the hippocampus and entorhinal cortex.

## 2. Installation
You can create a new environment using conda, and the yml file with all the right 
dependencies by running
```
conda env create --name NPG --file=requirements.yml
conda activate NPG
```

For now, install using pip for local editing and testing
```
pip install --upgrade setuptools wheel 
python setup.py sdist bdist_wheel
pip install -e .
```

## 3. Project

Try our package! We are gathering opinions to focus our efforts on improving aspects of the code, or adding new features, so if you tell us what you would like to have we might just implement it ;) Please refer to the [Road map](https://github.com/ClementineDomine/NeuralPlayground/blob/main/documents/road_map.md) to understand the state of the project and get an idea of the direction it is going in. This open-source software was built to be collaborative and lasting. We hope that the framework will be simple enough to be adopted by a great number of neuroscientists, eventually guiding the path to the computational understanding of the HEC mechanisms. We follow reproducible, inclusive, and collaborative project design guidelines. All relevant documents can be found in [Documents](https://github.com/ClementineDomine/NeuralPlayground/blob/main/documents/).

#### How to run a single model

You will find in the [Examples](https://github.com/ClementineDomine/NeuralPlayground/tree/main/examples) file a simple implementation of every single model in a single Environment. 

## 4. I-want-to-Contribute

There are many ways to contribute to the [NeuralPlayground](https://github.com/ClementineDomine/NeuralPlayground/tree/main/neuralplayground). 

 1. Implement a hippocampal and entorhinal cortex [Agent](https://github.com/ClementineDomine/NeuralPlayground/tree/main/neuralplayground/agents) of your choice. 
     
 2. Work on improving the [Arena](https://github.com/ClementineDomine/NeuralPlayground/tree/main/neuralplayground/arenas).
    
 3. Add an [Experiment](https://github.com/ClementineDomine/NeuralPlayground/tree/main/neuralplayground/experiments) al data set. 

All contributions should be submitted through a pull request that we will later access. 
Before sending a pull request, make sure you have the following: 

1. Checked the Licensing frameworks. 

2. Followed the [PEP8](https://www.python.org/dev/peps/pep-0008/) and [numpy docstring](https://numpydoc.readthedocs.io/en/latest/format.html). More details found in [Style Guide (https://github.com/ClementineDomine/NeuralPlayground/tree/main/documents/style_guide/style_guide.md).

3. Implemented and ran [Test](https://github.com/ClementineDomine/NeuralPlayground/tree/main/neuralplayground/tests).

4. Commented your work. 
    
All contributions to the repository are acknowledged through the all-contributors bot and in a future publication.
Refer to the README.md files found in each of the modules for further details on how to contribute to them.


### 5. Cite 

See [Citation](https://github.com/ClementineDomine/NeuralPlayground/blob/main/documents/citation.cff) for the correct citation of this framework. 

### 6. Lisence

More details about the liscence can be found at [Lisence](https://github.com/ClementineDomine/NeuralPlayground/blob/main/documents/lisence.md).


## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center"><a href="https://github.com/ClementineDomine"><img src="https://avatars.githubusercontent.com/u/18595111?v=4?s=100" width="100px;" alt="Clementine Domine"/><br /><sub><b>Clementine Domine</b></sub></a><br /><a href="#design-ClementineDomine" title="Design">🎨</a> <a href="#mentoring-ClementineDomine" title="Mentoring">🧑‍🏫</a> <a href="https://github.com/ClementineDomine/NeuralPlayground/commits?author=ClementineDomine" title="Code">💻</a> <a href="#data-ClementineDomine" title="Data">🔣</a></td>
      <td align="center"><a href="https://github.com/rodrigcd"><img src="https://avatars.githubusercontent.com/u/22643681?v=4?s=100" width="100px;" alt="rodrigcd"/><br /><sub><b>rodrigcd</b></sub></a><br /><a href="#design-rodrigcd" title="Design">🎨</a> <a href="#mentoring-rodrigcd" title="Mentoring">🧑‍🏫</a> <a href="https://github.com/ClementineDomine/NeuralPlayground/commits?author=rodrigcd" title="Code">💻</a> <a href="#data-rodrigcd" title="Data">🔣</a></td>
      <td align="center"><a href="https://github.com/LukeHollingsworth"><img src="https://avatars.githubusercontent.com/u/93782020?v=4?s=100" width="100px;" alt="Luke Hollingsworth"/><br /><sub><b>Luke Hollingsworth</b></sub></a><br /><a href="https://github.com/ClementineDomine/NeuralPlayground/commits?author=LukeHollingsworth" title="Documentation">📖</a> <a href="https://github.com/ClementineDomine/NeuralPlayground
      /commits?author=LukeHollingsworth" title="Code">💻</a></td>
      <td align="center"><a href="http://saxelab.org"><img src="https://avatars.githubusercontent.com/u/4165949?v=4?s=100" width="100px;" alt="Andrew Saxe"/><br /><sub><b>Andrew Saxe</b></sub></a><br /><a href="#mentoring-asaxe" title="Mentoring">🧑‍🏫</a></td>
      <td align="center"><a href="https://github.com/DrCaswellBarry"><img src="https://avatars.githubusercontent.com/u/17472149?v=4?s=100" width="100px;" alt="DrCaswellBarry"/><br /><sub><b>DrCaswellBarry</b></sub></a><br /><a href="#mentoring-DrCaswellBarry" title="Mentoring">🧑‍🏫</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
