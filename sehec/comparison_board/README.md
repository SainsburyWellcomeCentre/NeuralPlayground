# Environement: The standardised environment for the hippocampus and entorhinal cortex models. 

* [1 Introduction](#1-Introduction)
* [2 Arena Implemented](#2-Arena-Implemented)
* [3 How to Contribute](#3-How-to-Contribute)

## 1.Introduction

The environment aims at replicating the experimental setting where behavioral data and possible recordings werecollected.  Each specific environment implemented to resemble an experimental setting should be created as a subclassof the main environment class. We begin with
creating 2-dimensional arenas. The arena are build using unite walls that can be place in the two dimensional
space, delimiting the allowed transition of the Agent. This way, complex architectures such as a T-maze or a cycle,
resembling experimental settings can be created. In practice, the begging and end points of the
barriers are given as input. As a result the transition across line created by these two points are forbidden. The
Environment can be initialised with data from the real-life experiments. We will work toward improving each of
the Environments through the projects, adding experimental specifications

## 2.Arena-Implemented

1. Any 2D Environmement
2. Merging Room

## 3. How-to-Contribute

1. Create an environment class to following the template shown in the [Env_Core](https://github.com/ClementineDomine/EHC_model_comparison/blob/main/sehec/arena_and_exp/env_core.py) and the [Style Guide](https://github.com/ClementineDomine/EHC_model_comparison/tree/main/documents/style_guide). 

2. Record your contribution.

All contribution should be sumbited through a pull request that we will later acess. 
Before sending a pull request make sure you have: 

1. Checked the Lisencing frameworks. 

2. Followed the [Style Guide](https://github.com/ClementineDomine/EHC_model_comparison/tree/main/documents/style_guide).

3. Implemented and ran [Test](https://github.com/ClementineDomine/EHC_model_comparison/tree/main/sehec/tests).

4. Commented your work 
    
All contributions to the repository are acknowledged through the all-contributors bot and in future publicaiton.

