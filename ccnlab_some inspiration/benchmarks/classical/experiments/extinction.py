import pandas as pd
import ccnlab.benchmarks.classical.core as cc


@cc.registry.register
class Extinction_ContinuousVsPartial(cc.ClassicalConditioningExperiment):
  """When CS-US pairings are followed by presentations of CS alone or unpaired CS and US, the CR decreases. Partial reinforcement leads to slower extinction and a higher conditioning asymptote.

  Source: 2.1, 2,2 - Figure 6
  """
  def __init__(self, n_train_continuous=10, n_train_partial=5, n_extinction=16):
    super().__init__({
      'continuous':
        cc.seq(
          cc.seq(
            cc.trial('A+'),
            repeat=n_train_continuous,
            name='train',
          ), cc.seq(
            cc.trial('A-'),
            repeat=n_extinction,
            name='extinction',
          )
        ),
      'partial':
        cc.seq(
          cc.seq(
            cc.trial('A+'),
            cc.trial('A-'),
            repeat=n_train_partial,
            name='train',
          ), cc.seq(
            cc.trial('A-'),
            repeat=n_extinction,
            name='extinction',
          )
        ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='suppression [(ctx-cs)/ctx]',
      citation='Wagner et al. (1967)',
      animal='rat',
      cs='visual, auditory',
      us='shock',
      response='bar pressing',
      preparation='fear conditioning',
    )
    self.empirical_results = pd.melt(
      pd.DataFrame(
        columns=['group', 'session', 'A'],
        data=[
          ['continuous', 0, 1],
          ['continuous', 1, 0.95],
          ['continuous', 2, 0.75],
          ['continuous', 3, 0.25],
          ['continuous', 4, 0.05],
          ['partial', 0, 1],
          ['partial', 1, 0.95],
          ['partial', 2, 0.9],
          ['partial', 3, 0.7],
          ['partial', 4, 0.4],
        ]
      ),
      id_vars=['group', 'session']
    )
    self.plots = [
      lambda df, ax, **kwargs: cc.
      plot_lines(df, ax=ax, x='session', xlabel=kwargs['xlabel'], ylabel=kwargs['ylabel'])
    ]

  def simulated_results(self):
    return cc.trials_to_sessions(
      pd.melt(
        self.dataframe(
          lambda x: {
            'A': cc.conditioned_response(x['timesteps'], x['responses'], ['A']),
          } if x['phase'] == 'extinction' else None,
          include_trial=False,
          include_trial_in_phase=True,
        ),
        id_vars=['group', 'trial in phase']
      ).groupby(['group', 'trial in phase', 'variable'], sort=False).mean().reset_index(),
      4,
      trial_name='trial in phase',
      keep_first=True
    )
