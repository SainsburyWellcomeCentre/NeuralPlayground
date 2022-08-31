# Agent: The standardised environment for the hippocampus and entorhinal cortex models.


* [1. Introduction](#1-Introduction)
* [2. Model Implemented  ](#2-ModelImplemented )
* [3. How to Contribute](#5-HowtoContribute)

## 1.Introduction

The agent can be thought as the animal performing the task in the experiment.  All agent types will be given a set ofabilities that are summarised in the agent’s main class.  Each different model developed can be easily implemented asa subclass of the main one.  The agent class has inbuilt methods to interact with the environment, where the specificprocess might vary from model to model, but the overall structure is the same.  Multiple variables can be given, fromthe environment to the agent, depending on the needs of the model. Some examples are position, velocity, the positionof the reward, visual cues, head direction, whiskers, smell, among others.  Roughly, the agent and environment classesfollows the structure in OpenAI Gym and DeepMind Lab, along with additional specific methods, such asgetneural responsefor agents, andplot experimental resultsfor environments.


## 2.Model Implemented 

  1. [The Tolman-Eichenbaum machine](https://github.com/ClementineDomine/EHC_model_comparison/blob/main/sehec/agent/TEM)
  
 
  2. [The hippocampus as a predictive map](https://github.com/ClementineDomine/EHC_model_comparison/blob/main/sehec/agent/SRKim.py)

  4. [Learning place cells, grid cells and invariances with excitatory and inhibitory plasticity](https://github.com/ClementineDomine/EHC_model_comparison/blob/main/sehec/agent/weber_and_sprekeler.py)

Each of the models are implemented in a jupyter notebook in  [Examples](https://github.com/ClementineDomine/EHC_model_comparison/tree/main/examples) to facilitate the intereraction.

## 3.How to Contribute

  1. Create a file that indicates the appropirate reference to the model

  2. Create a class to run the model following the template shown in the [modelcore.py](https://github.com/ClementineDomine/EHC_model_comparison/blob/main/agent/core.py) and the [Style Guide](https://github.com/ClementineDomine/EHC_model_comparison/tree/main/Documents).
  
  3. Create a [Examples](https://github.com/ClementineDomine/EHC_model_comparison/tree/main/examples)  jupyter notebook for the new model where you can run the model in a chosen environement with selected experimental data
  
  3. Record your contribution

All contribution should be sumbited through a pull request that we will later acess. 
Before sending a pull request make sure you have:
1. Checked the Lisencing frameworks. 

2. Followed the [Style Guide](https://github.com/ClementineDomine/EHC_model_comparison/tree/main/Documents).

3. Implemented and ran [Test](https://github.com/ClementineDomine/EHC_model_comparison/tree/main/sehec/test).

4. Commented your work 
        
All contributions to the repository are acknowledged through the all-contributors bot and in future publicaiton.


