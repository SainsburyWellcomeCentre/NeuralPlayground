import pandas as pd
import ccnlab.benchmarks.classical.core as cc


@cc.registry.register
class Competition_RelativeValidity(cc.ClassicalConditioningExperiment):
  """Conditioning to X is weaker when training consists of pairing X with stimuli A/B that are
  correlated with reinforcement, than when training consists of pairing X with stimuli A/B that
  are not correlated. 

  Source: 7.1 - Figure 29
  """
  def __init__(self, n_correlated=200, n_uncorrelated=100, n_test=10):
    super().__init__({
      'correlated':
        cc.seq(
          cc.seq(
            cc.trial('XA+'),
            cc.trial('XB-'),
            repeat=n_correlated,
            name='train',
          ), cc.seq(
            cc.trial('X'),
            repeat=n_test,
            name='test',
          )
        ),
      'uncorrelated':
        cc.seq(
          cc.seq(
            cc.trial('XA+'),
            cc.trial('XA-'),
            cc.trial('XB+'),
            cc.trial('XB-'),
            repeat=n_uncorrelated,
            name='train',
          ), cc.seq(
            cc.trial('X'),
            repeat=n_test,
            name='test',
          )
        ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='conditioned response [%]',
      citation='Wagner et al. (1968)',
      animal='rat',
      cs='visual, auditory',
      us='food',
      response='bar pressing',
      preparation='appetitive conditioning',
    )
    self.empirical_results = pd.melt(
      pd.DataFrame(columns=['group', 'X'], data=[
        ['correlated', 20],
        ['uncorrelated', 80],
      ]),
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
          'X': cc.conditioned_response(x['timesteps'], x['responses'], ['X']),
        } if x['phase'] == 'test' else None,
        include_trial=False,
      ),
      id_vars=['group']
    ).groupby(['group', 'variable'], sort=False).mean().reset_index()


@cc.registry.register
class Competition_OvershadowingAndForwardBlocking(cc.ClassicalConditioningExperiment):
  """Training AB+ results in weaker conditioning to A than training A+ alone (overshadowing).
  Training B+ -> AB+ results in even weaker conditioning to A (forward blocking).

  Source: 7.2, 7.5 - Figure 30
  """
  def __init__(self, n_train=20, n_test=1):
    super().__init__({
      'control':
        cc.seq(
          cc.seq(cc.trial('-'), repeat=n_train, name='train'),
          cc.seq(cc.trial('A+'), repeat=n_train, name='train'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test'),
        ),
      'overshadowing':
        cc.seq(
          cc.seq(cc.trial('C+'), repeat=n_train, name='train'),
          cc.seq(cc.trial('AB+'), repeat=n_train, name='train'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test'),
        ),
      'forward blocking':
        cc.seq(
          cc.seq(cc.trial('B+'), repeat=n_train, name='train'),
          cc.seq(cc.trial('AB+'), repeat=n_train, name='train'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test'),
        ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='elevation scores',
      citation='Holland and Fox (2003)',
      animal='rat',
      cs='visual, auditory',
      us='food',
      response='feeding',
      preparation='appetitive conditioning',
    )
    self.empirical_results = pd.melt(
      pd.DataFrame(
        columns=['group', 'A'],
        data=[
          ['control', 65],
          ['overshadowing', 40],
          ['forward blocking', 12],
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
        } if x['phase'] == 'test' else None,
        include_trial=False,
      ),
      id_vars=['group']
    ).groupby(['group', 'variable'], sort=False).mean().reset_index()


@cc.registry.register
class Competition_Unblocking(cc.ClassicalConditioningExperiment):
  """In forward blocking B -> AB+, increasing or decreasing the US during AB presentation
  can increase responding to the blocked A.

  Source: 7.3, 7.4 - Figure 31
  """
  def __init__(self, n_first=40, n_second=280, n_test=1):
    super().__init__({
      'weak/weak':
        cc.seq(
          cc.seq(cc.trial('B+'), repeat=n_first, name='train'),
          cc.seq(cc.trial('AB+'), repeat=n_second, name='train'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test'),
        ),
      'weak/strong':
        cc.seq(
          cc.seq(cc.trial('B+'), repeat=n_first, name='train'),
          cc.seq(cc.trial('AB#'), repeat=n_second, name='train'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test'),
        ),
      'strong/strong':
        cc.seq(
          cc.seq(cc.trial('B#'), repeat=n_first, name='train'),
          cc.seq(cc.trial('AB#'), repeat=n_second, name='train'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test'),
        ),
      'strong/weak':
        cc.seq(
          cc.seq(cc.trial('B#'), repeat=n_first, name='train'),
          cc.seq(cc.trial('AB+'), repeat=n_second, name='train'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test'),
        ),
    })
    self.meta = dict(
      ylabel='suppression ratio',
      ydetail='suppression ratio',
      citation='Dickinson et al. (1976)',
      animal='rat',
      cs='visual, auditory',
      us='shock',
      response='bar pressing',
      preparation='fear conditioning',
    )
    self.empirical_results = pd.melt(
      pd.DataFrame(
        columns=['group', 'A'],
        data=[
          ['weak/weak', 0.47],
          ['weak/strong', 0.32],
          ['strong/strong', 0.44],
          ['strong/weak', 0.30],
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
          'A': cc.suppression_ratio(x['timesteps'], x['responses'], ['A']),
        } if x['phase'] == 'test' else None,
        include_trial=False,
      ),
      id_vars=['group']
    ).groupby(['group', 'variable'], sort=False).mean().reset_index()


@cc.registry.register
class Competition_BackwardBlocking(cc.ClassicalConditioningExperiment):
  """Training AB+ -> B+ results in weaker conditioning to A than training A+ alone (backward
  blocking).

  Source: 7.7 - Figure 33
  """
  def __init__(self, n_train=20, n_test=1):
    super().__init__({
      'control':
        cc.seq(
          cc.seq(cc.trial('AB+'), repeat=n_train, name='train'),
          cc.seq(cc.trial('C+'), repeat=n_train, name='train'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test'),
        ),
      'backward blocking':
        cc.seq(
          cc.seq(cc.trial('AB+'), repeat=n_train, name='train'),
          cc.seq(cc.trial('B+'), repeat=n_train, name='train'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test'),
        ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='mean latency [log s]',
      citation='Miller and Matute (1996)',
      animal='rat',
      cs='auditory',
      us='shock',
      response='drinking',
      preparation='fear conditioning',
    )
    self.empirical_results = pd.melt(
      pd.DataFrame(columns=['group', 'A'], data=[
        ['control', 1.55],
        ['backward blocking', 1.05],
      ]),
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
        } if x['phase'] == 'test' else None,
        include_trial=False,
      ),
      id_vars=['group']
    ).groupby(['group', 'variable'], sort=False).mean().reset_index()


@cc.registry.register
class Competition_Overexpectation(cc.ClassicalConditioningExperiment):
  """Training A+ -> B+ -> AB+ results in lower conditioning to A than without the AB+ compound.

  Source: 7.8 - Figure 34
  """
  def __init__(self, n_first=16, n_second=2, n_test=2):
    super().__init__({
      'control 1':
        cc.seq(
          cc.seq(cc.trial('A+'), repeat=n_first, name='train'),
          cc.seq(cc.trial('B+'), repeat=n_first, name='train'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test'),
        ),
      'control 2':
        cc.seq(
          cc.seq(cc.trial('A+'), repeat=n_first, name='train'),
          cc.seq(cc.trial('B+'), repeat=n_first, name='train'),
          cc.seq(cc.trial('A+'), repeat=n_second, name='train'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test'),
        ),
      'overexpectation':
        cc.seq(
          cc.seq(cc.trial('A+'), repeat=n_first, name='train'),
          cc.seq(cc.trial('B+'), repeat=n_first, name='train'),
          cc.seq(cc.trial('AB+'), repeat=n_second, name='train'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test'),
        ),
    })
    self.meta = dict(
      ylabel='suppression ratio',
      ydetail='suppression ratio',
      citation='Rescorla (1970)',
      animal='rat',
      cs='visual, auditory',
      us='shock',
      response='bar pressing',
      preparation='fear conditioning',
    )
    self.empirical_results = pd.melt(
      pd.DataFrame(
        columns=['group', 'A'],
        data=[
          ['control 1', 0.14],
          ['control 2', 0.11],
          ['overexpectation', 0.44],
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
          'A': cc.suppression_ratio(x['timesteps'], x['responses'], ['A']),
        } if x['phase'] == 'test' else None,
        include_trial=False,
      ),
      id_vars=['group']
    ).groupby(['group', 'variable'], sort=False).mean().reset_index()


@cc.registry.register
class Competition_Superconditioning(cc.ClassicalConditioningExperiment):
  """Training B- -> AB+ (superconditioning) results in higher conditioning to A than training AB+
  only (overshadowing) and yet higher than training B+ -> AB+ (forward blocking).

  Source: 7.9 - Figure 35
  """
  def __init__(self, n_first=40, n_second=2, n_test=3):
    super().__init__({
      'forward blocking':
        cc.seq(
          cc.seq(cc.trial('B+'), repeat=n_first, name='train'),
          cc.seq(cc.trial('AB+'), repeat=n_second, name='train'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test'),
        ),
      'overshadowing':
        cc.seq(
          cc.seq(cc.trial('-'), cc.trial('+'), repeat=n_first, name='train'),
          cc.seq(cc.trial('AB+'), repeat=n_second, name='train'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test'),
        ),
      'superconditioning':
        cc.seq(
          cc.seq(cc.trial('B-'), cc.trial('+'), repeat=n_first, name='train'),
          cc.seq(cc.trial('AB+'), repeat=n_second, name='train'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test'),
        ),
    })
    self.meta = dict(
      ylabel='suppression ratio',
      ydetail='suppression ratio',
      citation='Rescorla (1971)',
      animal='rat',
      cs='visual, auditory',
      us='shock',
      response='bar pressing',
      preparation='fear conditioning',
    )
    self.empirical_results = pd.melt(
      pd.DataFrame(
        columns=['group', 'A'],
        data=[
          ['forward blocking', 0.31],
          ['overshadowing', 0.25],
          ['superconditioning', 0.16],
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
          'A': cc.suppression_ratio(x['timesteps'], x['responses'], ['A']),
        } if x['phase'] == 'test' else None,
        include_trial=False,
      ),
      id_vars=['group']
    ).groupby(['group', 'variable'], sort=False).mean().reset_index()
