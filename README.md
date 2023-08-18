[![License](https://img.shields.io/badge/License-MIT-orange.svg)](https://opensource.org/licenses/MIT)
![CI](https://img.shields.io/github/actions/workflow/status/SainsburyWellcomeCentre/NeuralPlayground/test_and_deploy.yml?label=CI)
[![codecov](https://codecov.io/gh/SainsburyWellcomeCentre/NeuralPlayground/branch/main/graph/badge.svg?token=P8CCH3TI8K)](https://codecov.io/gh/SainsburyWellcomeCentre/NeuralPlayground)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v0.json)](https://github.com/charliermarsh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-9-orange.svg?style=flat-square)](#contributors)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

# NeuralPlayground 
**The  standardised environment for the hippocampus and entorhinal cortex models.** <a href="https://githubtocolab.com/SainsburyWellcomeCentre/NeuralPlayground/blob/main/examples/colab_example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<img src="images/NPG_GH-social-preview_white-bg.jpg" alt="NeuralPlayground Logo" width="500"/>

<!-- TOC -->

- [NeuralPlayground](#neuralplayground)
  - [Introduction](#introduction)
  - [Installation](#installation)
    - [Create a conda environment](#create-a-conda-environment)
    - [Pip install](#pip-install)
    - [Install for development](#install-for-development)
  - [Usage](#usage)
      - [How to run a single module](#how-to-run-a-single-module)
      - [How to run interactions between modules](#how-to-run-interactions-between-modules)
      - [How to run comparisons](#how-to-run-comparisons)
  - [I want to contribute](#i-want-to-contribute)
  - [Cite](#cite)
  - [License](#license)
  - [Contributors](#contributors)

<!-- /TOC -->

## Introduction
The abstract representation of space has been extensively studied in the hippocampus and entorhinal cortex in part 
due to the easy monitoring of the task and neural recording. A growing variety of theoretical models have been proposed
to capture the rich neural and behavioral phenomena associated with these circuits. However, objective comparison 
of these theories against each other and against empirical data is difficult.

Although the significance of virtuous interaction between experiments and theory is widely recognized, the tools 
available to facilitate comparison are limited. Some important challenge we aim to solve are:

   1. Lack availability and accessibility of data in a standardized, labeled format.

   2. Standard or easy ways to compare model output with empirical data.

   3. A repository of models and data sets that are relevant for the field of hippocampus and entorhinal cortex.

To address this gap, we present an open-source standardised software framework - NeuralPlayground - to enable 
comparison between the hippocampus and entorhinal cortex models. This Python software package offers a reproducible 
way to compare models against a centralised library of published experimental results, including neural recordings 
and animal behavior. The framework currently contains implementations of three Agents, as well as three Experiments 
providing simple interfaces to publicly available neural and behavioral data. It also contains a customizable 
2-dimensional Arena (continuous and discrete) able to produce common experimental environments in which the agents can 
move in and interact with. We note that each module can also be used separately, allowing flexible access to influential
models and data sets.

We currently rely on visual comparison of a hand-selected number of outputs of the model with neural recordings as 
shown in [github.com/NeuralPlayground/examples/comparison](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/blob/main/examples/comparisons_examples/comparison_from_manager.ipynb). In the future, a set of quantitative measures and 
qualitative measures will be added for systematic comparisons from any Agent, Arena, Experiments. 
We want to restate that this won't constitute an objective judgment of the quality of an Agent to replicate the brain 
mechanism. Instead, this allows an objective and complete comparison to the current evidence in the field, as is 
typically done in publications.

Altogether, we hope our framework, offers
a foundation that the community will build upon, working toward a shared, standardized, open, and
reproducible computational understanding of the hippocampus and entorhinal cortex.

Try our short tutorial online in Colab. <a href="https://githubtocolab.com/SainsburyWellcomeCentre/NeuralPlayground/blob/main/examples/colab_example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Installation

### Create a conda environment
We advise you to install the package in a virtual environment,
to avoid conflicts with other packages. For example, using `conda`:

```python
conda create --name NPG-env python=3.10
conda activate NPG-env
conda install pip
```

### Pip install

You can use `pip` get the latest release of NeuralPlayground from PyPI.
```python
# install the latest release
pip install NeuralPlayground

# upgrade to the latest release
pip install -U NeuralPlayground

# install a particular release
pip install NeuralPlayground==0.0.5
```
Note: if you wish to run our implementation of the Tolman-Eichenbaum machine, there are additional dependencies to install. These can be found in the [TEM_README.md](/examples/agent_examples/TEM_README.md) file.

### Install for development

If you want to contribute to the project, get the latest development version
from GitHub, and install it in editable mode, including the "dev" dependencies:

```bash
git clone https://github.com/SainsburyWellcomeCentre/NeuralPlayground/ --single-branch
cd NeuralPlayground
pip install -e .[dev]
```
> **Note**
> 
> if you are using the `zsh` shell (default on macOS), replace the last command with:
> ```bash
> pip install -e '.[dev]'
> ```

## Usage
Try our package! We are gathering opinions to focus our efforts on improving aspects of the code or adding new features,
so if you tell us what you would like to have, we might just implement it 😊. 
This open-source software was built to be collaborative and lasting. We hope that the framework will be simple 
enough to be adopted by a great number of neuroscientists, eventually guiding the path to the computational 
understanding of the HEC mechanisms. We follow reproducible, inclusive, and collaborative project design guidelines. 
All relevant documents can be found in [Documents](https://github.com/ClementineDomine/NeuralPlayground/blob/main/documents/).

#### How to run a single module

Each module can be used separately to easily explore and analyze experimental data and better understand any 
implemented model. Additionally, different Arenas can be initialised with artificial architectures or with data from 
real-life experiments. We provide examples of module instantiation in the detailed jupyter notebooks found in 
[examples_experiment](https://github.com/ClementineDomine/NeuralPlayground/tree/main/examples/experimental_examples), [examples_arena](https://github.com/ClementineDomine/NeuralPlayground/tree/main/examples/arena_examples) and [examples_agents](https://github.com/ClementineDomine/NeuralPlayground/tree/main/examples/agent_examples).
<strong>Check our Tolman-Eichenbaum Machine Implementation 
[this branch](https://github.com/ClementineDomine/NeuralPlayground/tree/whittington_2020) (work in progress), you will also need to install [pytorch](https://pytorch.org/).</strong>

#### How to run interactions between modules

As shown in the jupyter notebooks [Examples_agent](https://github.com/ClementineDomine/NeuralPlayground/tree/main/examples/agent_examples), the Agent can interact with an Arena in a standard RL 
framework. The first step is to initialise an Agent and Arena of your choice. The Agent can be thought of as the 
animal performing the Experiment and the Arena as the experimental setting where the animal navigates and performs a 
task.

#### How to run comparisons

We have a backend module to run simulations and compare the results with experimental data, you can find it in 
[simulation_manager](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/blob/main/examples/comparisons_examples/simulation_manager.ipynb) 
notebook. To plot these results, go to [comparison_from_manager](https://github.com/SainsburyWellcomeCentre/NeuralPlayground/blob/main/examples/comparisons_examples/comparison_from_manager.ipynb) and [examples_comparison](https://github.com/ClementineDomine/NeuralPlayground/blob/main/examples/comparison_board_examples/comparison_board.ipynb). We show visual 
comparisons between results from agents running with experimental behavior and results from the real experiment.

## I want to contribute

There are many ways to contribute to our project.

1. Implement a hippocampal and entorhinal cortex [Agent](https://github.com/ClementineDomine/NeuralPlayground/tree/main/neuralplayground/agents) of your choice.

2. Work on improving the [Arena](https://github.com/ClementineDomine/NeuralPlayground/tree/main/neuralplayground/arenas).

3. Add an [Experimental](https://github.com/ClementineDomine/NeuralPlayground/tree/main/neuralplayground/experiments) data set.

4. Implementing [metrics](https://github.com/ClementineDomine/NeuralPlayground/tree/main/neuralplayground/comparison/metrics.py) to compare the output of the Agent with the experimental data.

5. Refactor the code to improve the readability and efficiency.

All contributions should be submitted through a pull request and will be reviewed by the maintainers.
Before sending a pull request, make sure you have the done following:

1. Checked the Licensing frameworks.

2. Used the [right development environment](#install-for-development). Make sure to initialise the pre-commit hooks with ```pre-commit install``` and run ```pre-commit run -a``` to format the code and check for errors.

3. Followed the [PEP8](https://www.python.org/dev/peps/pep-0008/) and [numpy docstring](https://numpydoc.readthedocs.io/en/latest/format.html) style convention. More details found in [Style Guide](https://github.com/ClementineDomine/NeuralPlayground/tree/main/documents/style_guide.md).

4. Implemented and ran [tests](https://github.com/ClementineDomine/NeuralPlayground/tree/main/neuralplayground/tests).

5. Comment your work.

All contributions to the repository are acknowledged through the all-contributors bot.
Refer to the README.md files found in each of the modules for further details on how to contribute to them.


## Cite

See [Citation](https://github.com/ClementineDomine/NeuralPlayground/blob/main/documents/citation.cff) for the correct citation of this framework.

## License

⚖️ [MIT](./LICENSE)


## Contributors

Thanks go to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

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
      <td align="center" valign="top" width="14.28%"><a href="http://nikosirmpilatze.com"><img src="https://avatars.githubusercontent.com/u/20923448?v=4?s=100" width="100px;" alt="Niko Sirmpilatze"/><br /><sub><b>Niko Sirmpilatze</b></sub></a><br /><a href="#infra-niksirbi" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-niksirbi" title="Maintenance">🚧</a> <a href="#tool-niksirbi" title="Tools">🔧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://adamltyson.com"><img src="https://avatars.githubusercontent.com/u/13147259?v=4?s=100" width="100px;" alt="Adam Tyson"/><br /><sub><b>Adam Tyson</b></sub></a><br /><a href="#maintenance-adamltyson" title="Maintenance">🚧</a> <a href="#infra-adamltyson" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/rhayman"><img src="https://avatars.githubusercontent.com/u/5619644?v=4?s=100" width="100px;" alt="rhayman"/><br /><sub><b>rhayman</b></sub></a><br /><a href="https://github.com/SainsburyWellcomeCentre/NeuralPlayground/commits?author=rhayman" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/JarvisDevon"><img src="https://avatars.githubusercontent.com/u/30460455?v=4?s=100" width="100px;" alt="Devon Jarvis"/><br /><sub><b>Devon Jarvis</b></sub></a><br /><a href="https://github.com/SainsburyWellcomeCentre/NeuralPlayground/commits?author=JarvisDevon" title="Documentation">📖</a> <a href="https://github.com/SainsburyWellcomeCentre/NeuralPlayground/commits?author=JarvisDevon" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind are welcome!
