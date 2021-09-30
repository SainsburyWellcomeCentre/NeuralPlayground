from abc import ABC, abstractmethod
import numpy as np
import math
import random


def sigmoid(x):
  return 1.0 / (1.0 + math.exp(-x))


class Model(ABC):
  """Abstract class for all models."""
  @abstractmethod
  def reset(self):
    """Erase all memory from the model."""

  @abstractmethod
  def act(self, cs, ctx, us, t):
    """Perform an action given current stimulus input 
    :param cs: conditioned stimulus
    :param ctx: context
    :param us: unconditioned stimulus
    :param t: timestep within trial
    :return: (un)conditioned response
    """


class ValueBasedModel(Model):
  """Abstract class for models that calculate value through an RL-style framework."""
  def act(self, cs, ctx, us, t):
    response = self._value(cs, ctx, us, t)
    return response

  @abstractmethod
  def _value(self, cs, ctx, us, t):
    """Return value for current stimulus."""
