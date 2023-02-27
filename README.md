# The  standardised environment for the hippocampus and entorhinal cortex models. 
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-5-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->



* [1 Introduction](#1-Introduction)
* [2 Installation ](#2-Installation)
* [3 The Project](#3-Project)
* [4 How to Contribute](#4-I-want-to-Contribute)
* [5 Cite ](#5-Cite)

## 1. Introduction

This framework will allow neuroscientists to easily contrast the different hippocampus and entorhinal cortex models against experimental observations. This new tool aims at standardising the methods of building and comparing models and reporting empirical evidence. The code is constructed of two main classes, an agent class, and an environment class. The agent can be thought of as the animal performing the experiment, and the environment is where the animal navigates and performs a task. These two classes are allowed to interact to reproduce the full experimental setting.

We create this framework with the perspective that the neuroscience community should be able to continue its development by increasing the databases of models and experimental results without our intervention. Therefore, the software is made easy to use, facilitating future growth. This new environment will revolutionize how theoretical models are proposed in neuroscience and push for easy access and implementation of new ideas.

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
Please refer to the [Road map](https://github.com/ClementineDomine/NeuralPlayground/blob/main/documents/road_map.md) to understand the state of the project and get an idea of the direction it is going in. This open-source software was built to be collaborative and lasting. We hope that the framework will be simple enough to be adopted by a great number of neuroscientists, eventually guiding the path to the computational understanding of the HEC mechanisms. We follow reproducible, inclusive, and collaborative project design guidelines. All relevant documents can be found in [Documents](https://github.com/ClementineDomine/NeuralPlayground/blob/main/documents/)

#### How to run a single model
You will find in the [Examples](https://github.com/ClementineDomine/NeuralPlayground/tree/main/examples) file a simple implementation of every single model in a single Environment. The Agent and Environment class can be initialised to use data from an experiment. These two classes are allowed to interact to reproduce the full experimental setting. 

## 4. I-want-to-Contribute

There are many ways to contribute to the [NeuralPlayground](https://github.com/ClementineDomine/NeuralPlayground/tree/main/NeuralPlayground). 

 1. Implement a hippocampal and entorhinal cortex model of your choice. [Agent](https://github.com/ClementineDomine/NeuralPlayground/tree/main/NeuralPlayground/agents)
    
 2. Add functionality to the Comparison Board to compare results to real experimental data. [Comparison Board](https://github.com/ClementineDomine/NeuralPlayground/tree/main/NeuralPlayground/comparison_board)
    
 3. Work on improving the arena [Arena](https://github.com/ClementineDomine/NeuralPlayground/tree/main/NeuralPlayground/arenas)
    
 4. Add an experimental data set [Experiment](https://github.com/ClementineDomine/NeuralPlayground/tree/main/NeuralPlayground/experiments)

All contributions should be submitted through a pull request that we will later access. 
Before sending a pull request, make sure you have the following: 

1. Checked the Licensing frameworks. 

2. Followed them [Style Guide](https://github.com/ClementineDomine/NeuralPlayground/tree/main/documents/style_guide).

3. Implemented and ran [Test](https://github.com/ClementineDomine/NeuralPlayground/tree/main/NeuralPlayground/tests).

4. Commented your work 
    
All contributions to the repository are acknowledged through the all-contributors bot and in a future publication.
Refer to the README.md files found in each of the modules for further details on how to contribute to them.


### 5. Cite 

See [Citation](https://github.com/ClementineDomine/NeuralPlayground/blob/main/documents/Citation.cff) for the correct citation of this framework. 

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
