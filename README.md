# The  standardised environment for the hippocampus and entorhinal cortex models. 
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-3-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->


* [1 Introduction](#1-Introduction)
* [2 The Project](#1-Project)
* [3 Installation ](#3-Installation)
* [3 How to Contribute](#3Contribute)

## 1 Introduction

This framework will allow neuroscientists to easily contrast the different hippocampus and entorhinal cortex models against experimental observations. This new tool aims at standardising the methods of building and comparing models as well as reporting empirical evidences. The code is constructed of two main classes, an agent class and an environment class.  The agent can be thought of as the animal performing the experiment, and the environment as the experimental the where the animal navigates and performs a task.  These two classes are allowed to interact to reproduce the full experimental setting

We create this framework with the perspective that the neuroscience community should be able to continue its development by increasing the databases of model and experimental results without our intervention. Therefore, the software is made easy to use, facilitating future growth. 
This new environment will revolutionize how the theoretical models are proposed in neuroscience and push for easy access and implementation of new ideas. 


## 2 Project

### 2.1 The ProjectMotivation:

After an in-depth review of the literature of the entorhinal and hippocampus complex (EHC) models  we made several observations.
The EHC is a system that is given a great number of roles. On the one hand, the hippocampus is known to be involved in the consolidation of episodic memory to semantic memory from the hippocampus to the neocortex, presumably through the mechanism of replay.
On the other hand, the ECH is famously known for its function in navigation through its ability to encode spatial information through a cognitive map of the environment. Therefore we observe two main types of models and evidence. Models interested in the replay consolidation mechanisms during tasks and models that resemble neural patterns and behavior, both phenomena observed during navigation tasks. It doesn't seem easy to reconcile these two functions of the hippocampus and entorhinal cortex. However, we believe that creating an environment that easily allows comparing both types of models will guide us on the right track. 

We pay particular attention to the comparisons that each proposed model makes against experimental results. We first notice that many models results do not necessarily arise from the replication of the experimental behavioral data. Moreover, each researcher design specific implementation of animal simulation or behavioral data processing, repeating vainly the same work where similar tasks or behavioral data are used. We also noticed that models results are generally compared to hand-selected experimental observations that confirm the model's predictions. In addition, we observe an evident lack of counter-evidence and indication of the regimes where the model might produce outcomes that differ from experimental observations.
Indeed, there does not seem to be a consensus on the choice of experimental evidence. Considering the difficulty accessing the validity of experimental neuroscience observations from the multitude of experimental settings and the difficulty of building a model that assumes an extensive range of phenomena, the lack of counter-evidence seems natural. However, we believe that a single environment that allows the comparison of a model against multiple experimental observations will alleviate these difficulties and push for the standardisation of designing and testing new models. 


Finally, we came across multiple coding styles and implementations, complicating their understanding and comparison. In addition to proposing an environment to compare theoretical models against experimental data, this also requires to propose a common code structure that is flexible enough to add new experimental results and models. Hence, researchers could add their contributions without re-designing the rest of the code.\\

The above reasons lead us to build this project.

### 2.2 Inspiration

Benchmarks in machine learning are very common and helpful to assess the performance of models. It is undoubtedly challenging to construct a measure as precise for neuroscience evidence. However, one can imagine an environment that allows the comparison of a model to many experimental observations, which will effectively act as a benchmark. This was attempted in the paper 'CCNLab: A Benchmarking Framework for Computational Cognitive Neuroscience.' We aim at creating a similar environment for hippocampus and entorhinal cortex models. In our case, we deviate from the benchmark approach since we might not be able to give an exact measure of \textit{performance} of the model to match experimental data due to the diversity of conditions under which experiments are done. Instead, we plan to automatically generate several metrics and plots for easy quantitative and qualitative comparison of the results from the model and the experimental data.

### 2.3 Our Solution:

We, therefore, propose a standardised environment to compare hippocampus and entorhinal cortex models to alleviate all of the above limitations. This environment is being implemented as a software written in Python.
This software will allow the theoretical neuroscientists to compare their model efficiently and consistently against the same set of experimental results. This project was built to be collaborative and lasting. We hope that the framework will be simple enough to be adopted by a great number of neuroscientists, eventually guiding the path to the computational understanding of the hippocampal mechanisms.
This project will push for standardisation of neuroscience as a whole. Every hippocampal model integrated into this framework will have the same core structure.Along the same lines, the methods and details of the experiment will be reported in a standardised manner. Finally, we think this framework will guide the theorist in building their models by clearly laying out the set of observations they need to replicate. 




## 3 Installation


## 4 Contribute

There are three main ways to contribute to the porject: 

    1. Implement an hippocampal and entorhinal cortex models of your choice.

    2. Compare its results to real experimental data.

    3. Work on improving the environment

Refere to the README.md files found in the files of intterest for further details on how to contibute to the project.
We inspire the Organisation of the Project from the Turing Way Guide lines. Before Adding to the project, please make sure to check the Lisencing frameworks.





## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>

    <td align="center"><a href="https://github.com/ClementineDomine"><img src="https://avatars.githubusercontent.com/u/18595111?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Clementine Domine</b></sub></a><br /><a href="#design-ClementineDomine" title="Design">🎨</a> <a href="#mentoring-ClementineDomine" title="Mentoring">🧑‍🏫</a></td>
    <td align="center"><a href="https://github.com/rodrigcd"><img src="https://avatars.githubusercontent.com/u/22643681?v=4?s=100" width="100px;" alt=""/><br /><sub><b>rodrigcd</b></sub></a><br /><a href="#design-rodrigcd" title="Design">🎨</a><a href="#mentoring-ClementineDomine" title="Mentoring">🧑‍🏫</a></td>
    <td align="center"><a href="https://github.com/LukeHollingsworth"><img src="https://avatars.githubusercontent.com/u/93782020?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Luke Hollingsworth</b></sub></a><br /><a href="https://github.com/ClementineDomine/EHC_model_comparison/commits?author=LukeHollingsworth" title="Documentation">📖</a></td>


  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
