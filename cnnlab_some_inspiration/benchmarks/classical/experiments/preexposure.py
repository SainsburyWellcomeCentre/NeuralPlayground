import pandas as pd
import ccnlab.benchmarks.classical.core as cc


@cc.registry.register
class PreExposure_LatentInhibitionVsPerceptualLearning(cc.ClassicalConditioningExperiment):
  """Pre-exposure of CS A- before A+ pairings can result in reduced responding (latent inhibition)
  or increased responding (perceptual learning) depending if the context is the same or different,
  respectively.

  Source: 8.1, 8.7 - Figure 38
  """
  def __init__(self, n_preexpose=20, n_test=10):
    super().__init__({
      'same ctx, no preexpose':
        cc.seq(
          cc.seq(cc.trial('B-', ctx='K1'), repeat=n_preexpose, name='preexpose'),
          cc.seq(cc.trial('A+', ctx='K1'), repeat=n_test, name='test'),
        ),
      'same ctx, preexpose':
        cc.seq(
          cc.seq(cc.trial('A-', ctx='K1'), repeat=n_preexpose, name='preexpose'),
          cc.seq(cc.trial('A+', ctx='K1'), repeat=n_test, name='test'),
        ),
      'different ctx, no preexpose':
        cc.seq(
          cc.seq(cc.trial('B-', ctx='K1'), repeat=n_preexpose, name='preexpose'),
          cc.seq(cc.trial('A+', ctx='K2'), repeat=n_test, name='test'),
        ),
      'different ctx, preexpose':
        cc.seq(
          cc.seq(cc.trial('A-', ctx='K1'), repeat=n_preexpose, name='preexpose'),
          cc.seq(cc.trial('A+', ctx='K2'), repeat=n_test, name='test'),
        ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='1 - mean errors',
      citation='Lubow, Rifkin, & Alek (1976)',
      animal='rat',
      cs='olfactory',
      us='food',
      response='feeding',
      preparation='appetitive conditioning',
    )
    self.empirical_results = pd.melt(
      pd.DataFrame(
        columns=['group', 'A'],
        data=[
          ['same ctx, no preexpose', 1 - 0.33],
          ['same ctx, preexpose', 1 - 0.91],
          ['different ctx, no preexpose', 1 - 0.69],
          ['different ctx, preexpose', 1 - 0.28],
        ]
      ),
      id_vars=['group']
    )
    self.plots = [
      lambda df, ax, **kwargs: cc.
      plot_bars(df, ax=ax, x='group', xlabel=kwargs['xlabel'], ylabel=kwargs['ylabel'], wrap=14)
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
class PreExposure_USPreExposure(cc.ClassicalConditioningExperiment):
  """Pre-exposure of US + before A+ pairings results in decreased responding.

  Source: 8.5 - Figure 41
  """
  def __init__(self, n_preexpose=70, n_test=40):
    super().__init__({
      'no preexpose':
        cc.seq(cc.seq(cc.trial('A+'), repeat=n_test, name='test'),),
      'preexpose':
        cc.seq(
          cc.seq(cc.trial('+'), repeat=n_preexpose, name='preexpose'),
          cc.seq(cc.trial('A+'), repeat=n_test, name='test'),
        ),
    })
    self.meta = dict(
      ylabel='suppression ratio',
      ydetail='suppression ratio',
      citation='Kamin (1961)',
      animal='rat',
      cs='auditory',
      us='shock',
      response='bar pressing',
      preparation='fear conditioning',
    )
    self.empirical_results = pd.melt(
      pd.DataFrame(columns=['group', 'A'], data=[
        ['no preexpose', 0.12],
        ['preexpose', 0.23],
      ]),
      id_vars=['group']
    )
    self.plots = [
      lambda df, ax, **kwargs: cc.
      plot_bars(df, ax=ax, x='group', xlabel=kwargs['xlabel'], ylabel=kwargs['ylabel'], wrap=14)
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
