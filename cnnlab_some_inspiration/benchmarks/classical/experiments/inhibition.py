import pandas as pd
import ccnlab.benchmarks.classical.core as cc


@cc.registry.register
class Inhibition_InhibitorExtinction(cc.ClassicalConditioningExperiment):
  """Inhibitory conditioning to X trained via A+ -> AX- is extinguished by AX+ presentations.

  Source: 5.3 - Figure 24
  """
  def __init__(self, n_train=40, n_extinction=30, n_test=1):
    super().__init__({
      'control':
        cc.seq(
          cc.seq(cc.trial('A+'), cc.trial('AX-'), repeat=n_train, name='train'),
          cc.seq(cc.trial('-'), cc.trial('-'), repeat=n_extinction, name='train'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test-A'),
          cc.seq(cc.trial('AX'), repeat=n_test, name='test-AX'),
        ),
      'extinction':
        cc.seq(
          cc.seq(cc.trial('A+'), cc.trial('AX-'), repeat=n_train, name='train'),
          cc.seq(cc.trial('A+'), cc.trial('AX+'), repeat=n_extinction, name='train'),
          cc.seq(cc.trial('A'), repeat=n_test, name='test-A'),
          cc.seq(cc.trial('AX'), repeat=n_test, name='test-AX'),
        ),
    })
    self.meta = dict(
      ylabel='suppression ratio',
      ydetail='suppression ratio',
      citation='Zimmer-Hart & Rescorla (1974)',
      animal='rat',
      cs='visual, auditory',
      us='shock',
      response='bar pressing',
      preparation='fear conditioning',
    )
    self.empirical_results = pd.melt(
      pd.DataFrame(
        columns=['group', 'A', 'AX'], data=[
          ['control', 0.05, 0.23],
          ['extinction', 0.01, 0.01],
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
        } if x['phase'] == 'test-A' else {
          'AX': cc.suppression_ratio(x['timesteps'], x['responses'], ['A', 'X']),
        } if x['phase'] == 'test-AX' else None,
        include_trial=False,
      ),
      id_vars=['group']
    ).groupby(['group', 'variable'], sort=False).mean().reset_index()
