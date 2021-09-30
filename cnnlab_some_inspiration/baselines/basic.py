import numpy as np
import random
from ccnlab.baselines.core import Model, ValueBasedModel


class RescorlaWagner(ValueBasedModel):
  def __init__(self, cs_dim, ctx_dim, alpha=0.3):
    super().__init__()
    self.alpha = alpha  # Learning rate.
    self.cs_dim = cs_dim
    self.ctx_dim = ctx_dim
    self.D = self.cs_dim + self.ctx_dim  # Stimulus dimensions: punctate and contextual cues.
    self.reset()

  def reset(self):
    self.w = np.zeros((self.D,))

  def _value(self, cs, ctx, us, t):
    x = np.array(cs + ctx)
    v = self.w.dot(x)  # Value before update.
    self._update(x=x, r=us)
    return v

  def _update(self, x, r):
    rpe = r - self.w.dot(x)  # Reward prediction error.
    self.w = self.w + self.alpha * rpe * x  # Weight update.


class TemporalDifference(ValueBasedModel):
  def __init__(self, cs_dim, ctx_dim, num_timesteps, alpha=0.3, gamma=0.98):
    super().__init__()
    self.alpha = alpha  # Learning rate.
    self.gamma = gamma  # Discount factor.
    self.cs_dim = cs_dim
    self.ctx_dim = ctx_dim
    self.T = num_timesteps
    self.D = self.cs_dim + self.ctx_dim  # Stimulus dimensions: punctate and contextual cues.
    self.reset()

  def reset(self):
    self.w = np.zeros((self.D * self.T,))
    self.last_x = np.zeros((self.D * self.T,))  # Previous input.
    self.last_r = 0  # Previous reward.

  def _value(self, cs, ctx, us, t):
    if t == 0:
      self.last_x = np.zeros((self.D * self.T,))  # No previous input at initial timestep.
    x = np.zeros((self.D * self.T,))
    x[t * self.D:(t + 1) * self.D] = cs + ctx  # Complete serial compound representation.
    v = self.w.dot(x)  # Value before update.
    self._update(x=x, r=us)
    if t + 1 == self.T:
      self._update(x=np.zeros((self.D * self.T,)), r=0)  # Perform update with the last seen input.
    return v

  def _update(self, x, r):
    # Update for the previous input, because we don't have access to the next input.
    last_rpe = self.last_r + self.gamma * self.w.dot(x) - self.w.dot(self.last_x)
    self.w = self.w + self.alpha * last_rpe * self.last_x  # Weight update.
    self.last_x = x
    self.last_r = r


class KalmanFilter(ValueBasedModel):
  def __init__(self, cs_dim, ctx_dim, tau2=0.01, sigma_r2=1, sigma_w2=1):
    super().__init__()
    self.cs_dim = cs_dim
    self.ctx_dim = ctx_dim
    self.D = self.cs_dim + self.ctx_dim  # Stimulus dimensions: punctate and contextual cues.
    self.tau2 = tau2  # Diffusion/transition variance.
    self.sigma_r2 = sigma_r2  # Noise variance.
    self.sigma_w2 = sigma_w2  # Prior variance.
    self.Q = self.tau2 * np.identity(self.D)  # Transition covariance.
    self.reset()

  def reset(self):
    self.w = np.zeros((self.D,))  # Mean weights.
    self.S = self.sigma_w2 * np.identity(self.D)  # Weight covariance.

  def _value(self, cs, ctx, us, t):
    x = np.array(cs + ctx)
    v = self.w.dot(x)  # Value before update.
    self._update(x=x, r=us)
    return v

  def _update(self, x, r):
    rpe = r - self.w.dot(x)  # Reward prediction error.
    S = self.S + self.Q  # Prior covariance.
    R = x.dot(S).dot(x) + self.sigma_r2  # Residual covariance.
    k = S.dot(x) / R  # Kalman gain.
    self.w = self.w + k * rpe  # Weight update.
    self.S = S - k.dot(x) * S  # Posterior covariance.


class RandomModel(Model):
  """Produces response with probability that changes linearly with each US."""
  def __init__(self, start=0.2, delta=0.1, min_prob=0.1, max_prob=0.9):
    self.prob = start
    self.start = start
    self.delta = delta
    self.min_prob = min_prob
    self.max_prob = max_prob

  def reset(self):
    self.prob = self.start

  def act(self, cs, ctx, us, t):
    if us > 0:
      self.prob = max(min(self.prob + self.delta, self.max_prob), self.min_prob)
      return 1
    if len(cs) > 0:
      return random.choices([1, 0], weights=[self.prob, 1 - self.prob])[0]
    return 0
