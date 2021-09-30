import math
import pandas as pd
import ccnlab.benchmarks.classical.core as cc


@cc.registry.register
class HigherOrder_SensoryPreconditioning(cc.ClassicalConditioningExperiment):
  """When AB- pairings are followed by A+ pairings, presentation of B may generate a response.

  Source: 11.1 - Figure 57
  """
  def __init__(self, n_nonreinforced=100, n_reinforced=20, n_test=1):
    super().__init__({
      'control':
        cc.seq(
          cc.seq(cc.trial('-'), repeat=n_nonreinforced, name='train'),
          cc.seq(cc.trial('A+'), repeat=n_reinforced, name='train'),
          cc.seq(cc.trial('B'), repeat=n_test, name='test'),
        ),
      'sensory preconditioning':
        cc.seq(
          cc.seq(cc.trial('AB-'), repeat=n_nonreinforced, name='train'),
          cc.seq(cc.trial('A+'), repeat=n_reinforced, name='train'),
          cc.seq(cc.trial('B'), repeat=n_test, name='test'),
        ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='conditioned response',
      citation='Brogden (1939)',
      animal='dog',
      cs='visual, auditory',
      us='shock',
      response='flexion',
      preparation='reflex conditioning',
    )
    self.empirical_results = pd.melt(
      pd.DataFrame(
        columns=['group', 'B'], data=[
          ['control', 0.5],
          ['sensory preconditioning', 9.5],
        ]
      ),
      id_vars=['group']
    )
    self.plots = [
      lambda df, ax, **kwargs: cc.
      plot_bars(df, ax=ax, x='group', xlabel=kwargs['xlabel'], ylabel=kwargs['ylabel'])
    ]

  def simulated_results(self):
    return pd.melt(
      self.dataframe(
        lambda x: {
          'B': cc.conditioned_response(x['timesteps'], x['responses'], ['B']),
        } if x['phase'] == 'test' else None,
        include_trial=False,
      ),
      id_vars=['group']
    ).groupby(['group', 'variable'], sort=False).mean().reset_index()


@cc.registry.register
class HigherOrder_SecondOrderConditioning(cc.ClassicalConditioningExperiment):
  """When A+ pairings are followed by AB- pairings, presentation of B may generate a response. The
  number of BA- pairings determines whether second-order conditining or conditioned inhibition is
  obtained.

  Source: 11.2, 11.3 - Figure 58
  """
  def __init__(self, n_reinforced=170, n_nonreinforced_few=1, n_nonreinforced_many=85, n_test=4):
    super().__init__({
      'control':
        cc.seq(
          cc.seq(cc.trial('AB+'), repeat=n_reinforced, name='train'),
          cc.seq(cc.trial('B'), repeat=n_test, name='test'),
        ),
      'interspersed few':
        cc.seq(
          cc.seq(cc.trial('A+'), repeat=math.ceil(n_reinforced / 2), name='train'),
          cc.seq(cc.trial('AB-'), repeat=n_nonreinforced_few, name='train'),
          cc.seq(cc.trial('A+'), repeat=n_reinforced - math.ceil(n_reinforced / 2), name='train'),
          cc.seq(cc.trial('B'), repeat=n_test, name='test'),
        ),
      'sequential few':
        cc.seq(
          cc.seq(cc.trial('A+'), repeat=n_reinforced, name='train'),
          cc.seq(cc.trial('AB-'), repeat=n_nonreinforced_few, name='train'),
          cc.seq(cc.trial('B'), repeat=n_test, name='test'),
        ),
      'interspersed many':
        cc.seq(
          cc.seq(
            cc.seq(cc.trial('A+'), repeat=math.ceil(n_reinforced / n_nonreinforced_many)),
            cc.seq(cc.trial('AB-'), repeat=1),
            repeat=n_nonreinforced_many,
            name='train',
          ),
          cc.seq(cc.trial('B'), repeat=n_test, name='test'),
        ),
      'sequential many':
        cc.seq(
          cc.seq(cc.trial('A+'), repeat=n_reinforced, name='train'),
          cc.seq(cc.trial('AB-'), repeat=n_nonreinforced_many, name='train'),
          cc.seq(cc.trial('B'), repeat=n_test, name='test'),
        ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='mean latency [log s]',
      citation='Yin et al. (1994)',
      animal='rat',
      cs='auditory',
      us='shock',
      response='drinking',
      preparation='fear conditioning',
    )
    self.empirical_results = pd.melt(
      pd.DataFrame(
        columns=['group', 'B'],
        data=[
          ['control', 1.0],
          ['interspersed few', 1.9],
          ['sequential few', 1.73],
          ['interspersed many', 0.93],
          ['sequential many', 0.8],
        ]
      ),
      id_vars=['group']
    )
    self.plots = [
      lambda df, ax, **kwargs: cc.
      plot_bars(df, ax=ax, x='group', xlabel=kwargs['xlabel'], ylabel=kwargs['ylabel'], wrap=11)
    ]

  def simulated_results(self):
    return pd.melt(
      self.dataframe(
        lambda x: {
          'B': cc.conditioned_response(x['timesteps'], x['responses'], ['B']),
        } if x['phase'] == 'test' else None,
        include_trial=False,
      ),
      id_vars=['group']
    ).groupby(['group', 'variable'], sort=False).mean().reset_index()
