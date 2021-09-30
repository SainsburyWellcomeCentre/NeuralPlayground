import pandas as pd
import ccnlab.benchmarks.classical.core as cc


@cc.registry.register
class Generalization_NovelVsInhibitor(cc.ClassicalConditioningExperiment):
  """Adding a novel stimulus C to a trained stimulus A results in a smaller decrease in CR than does adding a conditioned inhibitor X to a trained stimulus A.

  Source: 3.2 - Figure 8
  """
  def __init__(self, n=10):
    train = [
      cc.seq(
        cc.trial('A+'),
        cc.trial('B+'),
        repeat=n,
        name='train 1',
      ),
      cc.seq(
        cc.trial('A+'),
        cc.trial('B+'),
        cc.trial('BX-'),
        repeat=n,
        name='train 2',
      ),
    ]
    super().__init__({
      'control': cc.seq(
        *train,
        cc.seq(
          cc.trial('A'),
          repeat=n,
          name='test',
        ),
      ),
      'external inhibition': cc.seq(
        *train,
        cc.seq(
          cc.trial('AC'),
          repeat=n,
          name='test',
        ),
      ),
      'conditioned inhibition': cc.seq(
        *train,
        cc.seq(
          cc.trial('AX'),
          repeat=n,
          name='test',
        ),
      ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='ratings',
      citation='Kutlu & Schmajuk (2012)',
      animal='human',
      cs='visual',
      us='value',
      response='value prediction',
      preparation='value prediction',
    )
    self.empirical_results = pd.DataFrame(
      columns=['group', 'variable', 'value'],
      data=[
        ['control', 'A', 9],
        ['external inhibition', 'AC', 5.5],
        ['conditioned inhibition', 'AX', 2],
      ]
    )
    self.plots = [
      lambda df, ax, **kwargs: cc.plot_bars(
        df, ax=ax, x='group', split=None, xlabel=kwargs['xlabel'], ylabel=kwargs['ylabel'], wrap=11
      )
    ]

  def simulated_results(self):
    return pd.melt(
      self.dataframe(
        lambda x: {
          **({
            'A': cc.conditioned_response(x['timesteps'], x['responses'], ['A'])
          } if x['group'] == 'control' else {}),
          **({
            'AC': cc.conditioned_response(x['timesteps'], x['responses'], ['A', 'C'])
          } if x['group'] == 'external inhibition' else {}),
          **({
            'AX': cc.conditioned_response(x['timesteps'], x['responses'], ['A', 'X'])
          } if x['group'] == 'conditioned inhibition' else {}),
        } if x['phase'] == 'test' else None,
        include_trial=False,
      ),
      id_vars=['group']
    ).groupby(['group', 'variable'], sort=False).mean().dropna().reset_index()


@cc.registry.register
class Generalization_AddVsRemove(cc.ClassicalConditioningExperiment):
  """Adding a cue to a trained compound results in a smaller decrease in CR than does removing a
  cue from a trained compound.

  Source: 3.3 - Figure 9
  """
  def __init__(self, n_train=500, n_test=4):
    # We need to separate test phases for each variable being tested. Otherwise, e.g. for a test
    # trial that only tests A, we would count 0 for AB and ABC which were not presented.
    test = cc.seq(
      cc.seq(cc.trial('A'), name='test-A'),
      cc.seq(cc.trial('AB'), name='test-AB'),
      cc.seq(cc.trial('ABC'), name='test-ABC'),
      repeat=n_test,
    )
    super().__init__({
      'A': cc.seq(
        cc.seq(cc.trial('A+'), repeat=n_train, name='train'),
        test,
      ),
      'AB': cc.seq(
        cc.seq(cc.trial('AB+'), repeat=n_train, name='train'),
        test,
      ),
      'ABC': cc.seq(
        cc.seq(cc.trial('ABC+'), repeat=n_train, name='train'),
        test,
      ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='conditioned response [%]',
      citation='Brandon et al. (2000)',
      animal='rabbit',
      cs='visual, auditory, tactile',
      us='shock',
      response='eyeblink',
      preparation='eyeblink conditioning',
    )
    self.empirical_results = pd.melt(
      pd.DataFrame(
        columns=['group', 'A', 'AB', 'ABC'],
        data=[
          ['A', 80, 65, 55],
          ['AB', 20, 90, 70],
          ['ABC', 20, 50, 85],
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
          'A': cc.conditioned_response(x['timesteps'], x['responses'], ['A']),
        } if x['phase'] == 'test-A' else {
          'AB': cc.conditioned_response(x['timesteps'], x['responses'], ['A', 'B']),
        } if x['phase'] == 'test-AB' else {
          'ABC': cc.conditioned_response(x['timesteps'], x['responses'], ['A', 'B', 'C']),
        } if x['phase'] == 'test-ABC' else None,
        include_trial=False
      ),
      id_vars=['group']
    ).groupby(['group', 'variable'], sort=False).mean().reset_index()
