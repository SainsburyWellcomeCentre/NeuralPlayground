import pandas as pd
import ccnlab.benchmarks.classical.core as cc


@cc.registry.register
class Acquisition_ContinuousVsPartial(cc.ClassicalConditioningExperiment):
  """With repeated CS-US pairings, CS elicits CR that increases in magnitude and frequency with further reinforcement. Partial reinforcement leads to slower acquisition and a lower conditioning
  asymptote.

  Source: 1.1, 1.2 - Figure 2
  """
  def __init__(self, n=64, partial_prob=0.5):
    super().__init__({
      'continuous':
        cc.seq(
          cc.trial('A+'),
          repeat=n,
          name='train',
        ),
      'partial':
        cc.seq(
          cc.sample({
            cc.trial('A+'): partial_prob,
            cc.trial('A-'): 1 - partial_prob,
          }),
          repeat=n,
          name='train',
        ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='conditioned response',
      citation='Wagner et al. (1967)',
      animal='rat',
      cs='visual',
      us='shock',
      response='auditory startle',
      preparation='fear conditioning',
    )
    self.empirical_results = pd.melt(
      pd.DataFrame(
        columns=['group', 'session', 'A'],
        data=[
          ['continuous', 0, 0],
          ['continuous', 1, 14],
          ['continuous', 2, 17],
          ['continuous', 3, 18],
          ['continuous', 4, 17.5],
          ['partial', 0, 0],
          ['partial', 1, 10],
          ['partial', 2, 16],
          ['partial', 3, 13],
          ['partial', 4, 15],
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
        self.
        dataframe(lambda x: {
          'A': cc.conditioned_response(x['timesteps'], x['responses'], ['A']),
        }),
        id_vars=['group', 'trial']
      ).groupby(['group', 'trial', 'variable'], sort=False).mean().reset_index(),
      16,
      keep_first=True
    )
