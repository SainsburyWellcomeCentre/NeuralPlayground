# The  standardised environment for the hippocampus and entorhinal cortex models. (Experiment)


* [1 Introduction](#1-Introduction)
* [1 Contribute](#2-Contribute)

## 1 Introduction

We use open-source data of the experiment to replicate the experimental condition.
We use the following data sets:

1. “Conjunctive Representation of Position, Direction, and Velocity in Entorhinal Cortex” Sargolini et al. 2006 Conjuctive cells
2. “Hippocampus-independent phase precession in entorhinal grid cells”, Hafting et al 2008.
   
One of our goals is to expand this list to add more experiments that are relevant to this literature and that are publicly available. We hope to assess the performance of the models against a set of selected experimental observations, roughly categorized as qualitative and quantitative evidence.

1. Qualitative evidence: Types of cells: grid, place, band, presence of replay, phase precession, is the model biologically connected? Is the model biologically plausible? 
2. Quantitative evidence: Learning time scales, replay directions under different conditions, animal trajectories, among others.

The users of this software might only use a subset of the available experiments and comparisons depending on the capabilities of the proposed models. We want to reinstate that this won't constitute an objective judgment of the quality of a model to replicate the brain mechanism. Instead, this only allows an objective and complete comparison to the current evidence in the field.

## 2 Contribute

1. Create a file that indicates where to download the data.
2. Create a class to read/filter the data following the template shown in the [Behavior_data] and the [Style Guide](https://github.com/ClementineDomine/EHC_model_comparison/tree/main/Documents).
3. Cite the Data approrialty.
4. Use the data to be compared to the model results.
5. Record your contribution
