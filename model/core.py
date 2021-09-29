from abc import ABC, abstractmethod
import numpy as np
import math
import random

 """ Hope you have a good day of working on the RORO CLEM COLAB """
 
 
class NeuralResponseModel(ABC):
  """Abstract class for all models."""
  @abstractmethod
  def reset(self):
    """Erase all memory from the model."""

  @abstractmethod
  def neuralresponse(self, observation):
    """Perform an action given current stimulus input 
    observation dictionanary with:
        :param h: Head direction
        :param x: Position
        :param v: Velocity
        :param s: stimuli
        :param t: timestep within trial
        :param self: internal state
    return: neural response() (This might go to update self (state) state but this is unsure still)
        """
  @abstractmethod
  def act(self, observation):
      """Perform an action given current stimulus input
      observation dictionanary with:
        :param h: Head direction
        :param x: Position
        :param v: Velocity
        :param s:  stimuli
        :param t: timestep within trial
        :param self: internal state
    return: update/move(x,v,h,s) vector which will go to update the environment observation
        """


