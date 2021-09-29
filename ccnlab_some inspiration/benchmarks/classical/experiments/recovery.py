import pandas as pd
import ccnlab.benchmarks.classical.core as cc


@cc.registry.register
class Recovery_LatentInhibition(cc.ClassicalConditioningExperiment):
  """Extensive exposure to the context after training results in reduction of latent inhibition.

  Source: 10.1 - Figure 50
  """
  def __init__(self, n_nonreinforced=40, n_reinforced=10, n_recovery=40, n_test=3):
    super().__init__({
      'control':
        cc.seq(
          cc.seq(cc.trial('-'), repeat=n_nonreinforced, name='train'),
          cc.seq(cc.trial('A+'), repeat=n_reinforced, name='train'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test'),
        ),
      'latent inhibition':
        cc.seq(
          cc.seq(cc.trial('A-'), repeat=n_nonreinforced, name='train'),
          cc.seq(cc.trial('A+'), repeat=n_reinforced, name='train'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test'),
        ),
      'control recovery':
        cc.seq(
          cc.seq(cc.trial('-'), repeat=n_nonreinforced, name='train'),
          cc.seq(cc.trial('A+'), repeat=n_reinforced, name='train'),
          cc.seq(cc.trial('-'), repeat=n_recovery, name='train'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test'),
        ),
      'latent inhibition recovery':
        cc.seq(
          cc.seq(cc.trial('A-'), repeat=n_nonreinforced, name='train'),
          cc.seq(cc.trial('A+'), repeat=n_reinforced, name='train'),
          cc.seq(cc.trial('-'), repeat=n_recovery, name='train'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test'),
        ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='mean latency [log s]',
      citation='Grahame et al. (1994)',
      animal='rat',
      cs='auditory',
      us='shock',
      response='drinking',
      preparation='fear conditioning',
    )
    self.empirical_results = pd.melt(
      pd.DataFrame(
        columns=['group', 'A'],
        data=[
          ['control', 2.2],
          ['latent inhibition', 1.6],
          ['control recovery', 1.9],
          ['latent inhibition recovery', 1.9],
        ]
      ),
      id_vars=['group']
    )
    self.plots = [
      lambda df, ax, **kwargs: cc.
      plot_bars(df, ax=ax, x='group', xlabel=kwargs['xlabel'], ylabel=kwargs['ylabel'], wrap=9)
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
class Recovery_Overshadowing(cc.ClassicalConditioningExperiment):
  """Extinction of B after overshadowing training AB+ results in increased responding to A.

  Source: 10.2 - Figure 51
  """
  def __init__(self, n_reinforced=10, n_nonreinforced=10, n_test=1):
    super().__init__({
      'control':
        cc.seq(
          cc.seq(cc.trial('A+'), repeat=n_reinforced, name='train'),
          cc.seq(cc.trial('-'), repeat=n_nonreinforced, name='train'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test'),
        ),
      'overshadowing':
        cc.seq(
          cc.seq(cc.trial('AB+'), repeat=n_reinforced, name='train'),
          cc.seq(cc.trial('-'), repeat=n_nonreinforced, name='train'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test'),
        ),
      'overshadowing recovery':
        cc.seq(
          cc.seq(cc.trial('AB+'), repeat=n_reinforced, name='train'),
          cc.seq(cc.trial('B-'), repeat=n_nonreinforced, name='train'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test'),
        ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='mean latency [log s]',
      citation='Matzel et al. (1985)',
      animal='rat',
      cs='visual, auditory',
      us='shock',
      response='drinking',
      preparation='fear conditioning',
    )
    self.empirical_results = pd.melt(
      pd.DataFrame(
        columns=['group', 'A'],
        data=[
          ['control', 1.95],
          ['overshadowing', 1.05],
          ['overshadowing recovery', 1.55],
        ]
      ),
      id_vars=['group']
    )
    self.plots = [
      lambda df, ax, **kwargs: cc.
      plot_bars(df, ax=ax, x='group', xlabel=kwargs['xlabel'], ylabel=kwargs['ylabel'], wrap=13)
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
class Recovery_ExternalDisinhibition(cc.ClassicalConditioningExperiment):
  """Presenting a novel stimulus immediately before a previously extinguished CS might produce
  renewed responding.

  Source: 10.5 - Figure 53
  """
  def __init__(self, n_acquisition=10, n_extinction=30, n_test=3):
    super().__init__({
      'main':
        cc.seq(
          cc.seq(cc.trial('A+'), repeat=n_acquisition, name='acquisition'),
          cc.seq(cc.trial('A-'), repeat=n_extinction, name='extinction'),
          cc.seq(
            cc.trial(('B', 8, 12), ('A', 16, 20), ('+', 19, 20)), repeat=n_test, name='test-1'
          ),
          cc.seq(
            cc.trial(('B', 8, 12), ('A', 16, 20), ('+', 19, 20)), repeat=n_test, name='test-2'
          ),
        ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='median approach-withdrawal ratio',
      citation='Bottjer (1982)',
      animal='pigeon',
      cs='visual',
      us='food',
      response='feeding',
      preparation='appetitive conditioning',
    )
    self.empirical_results = pd.melt(
      pd.DataFrame(
        columns=['group', 'extinction', 'test 1', 'test 2'], data=[
          ['main', 0.5, 0.85, 0.78],
        ]
      ),
      id_vars=['group']
    )
    self.plots = [
      lambda df, ax, **kwargs: cc.
      plot_bars(df, ax=ax, x='group', xlabel=kwargs['xlabel'], ylabel=kwargs['ylabel'], wrap=13)
    ]

  def simulated_results(self):
    return pd.melt(
      self.dataframe(
        lambda x: {
          'extinction': cc.conditioned_response(x['timesteps'], x['responses'], ['A']),
        } if x['phase'] == 'extinction' else {
          'test 1': cc.conditioned_response(x['timesteps'], x['responses'], ['A']),
        } if x['phase'] == 'test-1' else {
          'test 2': cc.conditioned_response(x['timesteps'], x['responses'], ['A']),
        } if x['phase'] == 'test-2' else None,
        include_trial=False,
      ),
      id_vars=['group']
    ).groupby(['group', 'variable'], sort=False).mean().reset_index()


@cc.registry.register
class Recovery_SpontaneousRecovery(cc.ClassicalConditioningExperiment):
  """Presenting the CS some time after the subject has stopped responding might yield renewed
  responding.

  Source: 10.6 - Figure 54
  """
  def __init__(self, n_acquisition=12, n_extinction=4, n_delay=12, n_test=4):
    super().__init__({
      'no delay':
        cc.seq(
          cc.seq(cc.trial('A+', ctx='K1'), repeat=n_acquisition, name='acquisition'),
          cc.seq(cc.trial('A-', ctx='K1'), repeat=n_extinction, name='extinction'),
          cc.seq(cc.trial('A', ctx='K1'), repeat=n_test, name='test'),
        ),
      'delay':
        cc.seq(
          cc.seq(cc.trial('A+', ctx='K1'), repeat=n_acquisition, name='acquisition'),
          cc.seq(cc.trial('-', ctx='K2'), repeat=n_delay, name='delay'),
          cc.seq(cc.trial('A-', ctx='K1'), repeat=n_extinction, name='extinction'),
          cc.seq(cc.trial('A', ctx='K1'), repeat=n_test, name='test'),
        ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='conditioned response [#/min]',
      citation='Rescorla (2004)',
      animal='rat',
      cs='visual, auditory',
      us='food',
      response='feeding',
      preparation='appetitive conditioning',
    )
    self.empirical_results = pd.melt(
      pd.DataFrame(
        columns=['group', 'session', 'A'],
        data=[
          ['no delay', 1, 5],  # Acquisition
          ['no delay', 2, 8],
          ['no delay', 3, 10],
          ['no delay', 4, 11.5],
          ['no delay', 5, 13],
          ['no delay', 6, 13.2],
          ['no delay', 7, 14.5],
          ['no delay', 8, 14],
          ['no delay', 9, 14.2],
          ['no delay', 10, 14.2],
          ['no delay', 11, 14.6],
          ['no delay', 12, 15],
          ['no delay', 13, 8.5],  # Extinction
          ['no delay', 14, 3.8],
          ['no delay', 15, 3.2],
          ['no delay', 16, 1.4],
          ['no delay', 17, 0.2],  # Test
          ['no delay', 18, 0.1],
          ['no delay', 19, 0.2],
          ['no delay', 20, 0.8],
          ['delay', 1, 5],  # Acquisition
          ['delay', 2, 8],
          ['delay', 3, 10],
          ['delay', 4, 11.5],
          ['delay', 5, 13],
          ['delay', 6, 13.2],
          ['delay', 7, 14.5],
          ['delay', 8, 14],
          ['delay', 9, 14.2],
          ['delay', 10, 14.2],
          ['delay', 11, 14.6],
          ['delay', 12, 15],
          ['delay', 13, 10.6],  # Extinction
          ['delay', 14, 4],
          ['delay', 15, 2.4],
          ['delay', 16, 2],
          ['delay', 17, 6.2],  # Test
          ['delay', 18, 1.2],
          ['delay', 19, 2.2],
          ['delay', 20, 1.6],
        ]
      ),
      id_vars=['group', 'session']
    )
    self.plots = [
      lambda df, ax, **kwargs: cc.
      plot_lines(df, ax=ax, x='session', xlabel=kwargs['xlabel'], ylabel=kwargs['ylabel'])
    ]

  def simulated_results(self):
    # Manually reindex trials to remove 'delay' phase trials.
    dfs = {}
    for phase in ('acquisition', 'extinction', 'test'):
      dfs[phase] = cc.trials_to_sessions(
        pd.melt(
          self.dataframe(
            lambda x: {
              'A': cc.conditioned_response(x['timesteps'], x['responses'], ['A']),
            } if x['phase'] == phase else None,
            include_trial=False,
            include_trial_in_phase=True,
          ),
          id_vars=['group', 'trial in phase']
        ).groupby(['group', 'trial in phase', 'variable'], sort=False).mean().reset_index(),
        1,
        trial_name='trial in phase',
      )
    dfs['extinction']['session'] += 12
    dfs['test']['session'] += 16
    return pd.concat(dfs.values()).reset_index(drop=True)


@cc.registry.register
class Recovery_Renewal(cc.ClassicalConditioningExperiment):
  """After extinction, presentation of the CS in a novel context might yield renewed responding.

  Source: 10.7 - Figure 55
  """
  def __init__(self, n_acquisition=15, n_extinction=20, n_test=10):
    super().__init__({
      'same context':
        cc.seq(
          cc.seq(cc.trial('A+', ctx='K3'), repeat=n_acquisition, name='acquisition'),
          cc.seq(cc.trial('A-', ctx='K1'), repeat=n_extinction, name='extinction'),
          cc.seq(cc.trial('A', ctx='K1'), repeat=n_test, name='test'),
        ),
      'novel context':
        cc.seq(
          cc.seq(cc.trial('A+', ctx='K3'), repeat=n_acquisition, name='acquisition'),
          cc.seq(cc.trial('A-', ctx='K2'), repeat=n_extinction, name='extinction'),
          cc.seq(cc.trial('A', ctx='K1'), repeat=n_test, name='test'),
        ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='conditioned response [%]',
      citation='Harris et al. (2000)',
      animal='rat',
      cs='auditory',
      us='shock',
      response='freezing',
      preparation='fear conditioning',
    )
    self.empirical_results = pd.melt(
      pd.DataFrame(columns=['group', 'A'], data=[
        ['same context', 27],
        ['novel context', 55],
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
class Recovery_Reinstatement(cc.ClassicalConditioningExperiment):
  """After extinction, presentation of the US in the context might yield renewed responding.

  Source: 10.8 - Figure 56
  """
  def __init__(self, n_acquisition=100, n_extinction=35, n_reinstatement=40, n_test=10):
    super().__init__({
      'no US':
        cc.seq(
          cc.seq(cc.trial('A+'), repeat=n_acquisition, name='acquisition'),
          cc.seq(cc.trial('A-'), repeat=n_extinction, name='extinction'),
          cc.seq(cc.trial('-'), repeat=n_reinstatement, name='reinstatement'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test-1'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test-2'),
        ),
      'US':
        cc.seq(
          cc.seq(cc.trial('A+'), repeat=n_acquisition, name='acquisition'),
          cc.seq(cc.trial('A-'), repeat=n_extinction, name='extinction'),
          cc.seq(cc.trial('+'), repeat=n_reinstatement, name='reinstatement'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test-1'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test-2'),
        ),
    })
    self.meta = dict(
      ylabel='suppression ratio',
      ydetail='suppression ratio',
      citation='Rescorla & Heth (1975)',
      animal='rat',
      cs='auditory',
      us='shock',
      response='bar pressing',
      preparation='fear conditioning',
    )
    self.empirical_results = pd.melt(
      pd.DataFrame(
        columns=['group', 'extinction', 'test 1', 'test 2'],
        data=[
          ['no US', 0.35, 0.4, 0.4],
          ['US', 0.4, 0.21, 0.36],
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
          'extinction': cc.suppression_ratio(x['timesteps'], x['responses'], ['A']),
        } if x['phase'] == 'extinction' else {
          'test 1': cc.suppression_ratio(x['timesteps'], x['responses'], ['A']),
        } if x['phase'] == 'test-1' else {
          'test 2': cc.suppression_ratio(x['timesteps'], x['responses'], ['A']),
        } if x['phase'] == 'test-2' else None,
        include_trial=False,
      ),
      id_vars=['group']
    ).groupby(['group', 'variable'], sort=False).mean().reset_index()
