# The  standardised environment for the hippocampus and entorhinal cortex models. (AGENT )


* [1 Introduction](#1-Introduction)
* [2 Model Implemented  ](#2-ModelImplemented )
* [3 How to Contribute](#5-HowtoContribute)

## 1 Introduction

The agent can be thought as the animal performing the task in the experiment.  All agent types will be given a set ofabilities that are summarised in the agent’s main class.  Each different model developed can be easily implemented asa subclass of the main one.  The agent class has inbuilt methods to interact with the environment, where the specificprocess might vary from model to model, but the overall structure is the same.  Multiple variables can be given, fromthe environment to the agent, depending on the needs of the model. Some examples are position, velocity, the positionof the reward, visual cues, head direction, whiskers, smell, among others.  Roughly, the agent and environment classesfollows the structure in OpenAI Gym[2] and DeepMind Lab[3], along with additional specific methods, such asgetneural responsefor agents, andplot experimental resultsfor environments.


## 2 Model Implemented 

Here we list some of the models we reviewed to implement in the first version of the software.  As mentioned before,each model predicts a different set of experimental observations that we will need to organize comprehensively.  Fornow, a rough taxonomy for these models could bereplayandnavigationrelated models (navigation ones are moreconnected with the place and grid cells predictions)

• The Tolman-Eichenbaum machine 
• An oscillatory interference model of grid cell firing 
• A general model of hippocampal and dorsal striatal learning and decision making 
• Learning place cells, grid cells and invariances with excitatory and inhibitory plasticity

Some of the models are implemented in a jupyter notebook to facilitate the intereraction.

## 3 How to Contribute

Follow the The model core template and the style guide to incorportate your model.  
Finnaly make sure on has the the Lisencing 


