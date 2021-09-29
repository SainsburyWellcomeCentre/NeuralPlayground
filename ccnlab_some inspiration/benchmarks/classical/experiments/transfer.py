import pandas as pd
import ccnlab.benchmarks.classical.core as cc


@cc.registry.register
class Transfer_Reacquisition(cc.ClassicalConditioningExperiment):
  """Following acquistion A+ and extinction A-, A+ pairings can result in faster or slower
  reacquisition depending on the number of extinction trials.

  Source: 9.2 - Figure 48
  """
  def __init__(
    self,
    n_acquisition=10,
    n_extinction_few=15,
    n_extinction_many=100,
    n_reacquisition_few=8,
    n_reacquisition_many=12
  ):
    super().__init__({
      'control few':
        cc.seq(
          cc.seq(cc.trial('-'), repeat=n_acquisition, name='acquisition'),
          cc.seq(cc.trial('-'), repeat=n_extinction_few, name='extinction'),
          cc.seq(cc.trial('A+'), repeat=n_reacquisition_few, name='reacquisition'),
        ),
      'extinction few':
        cc.seq(
          cc.seq(cc.trial('A+'), repeat=n_acquisition, name='acquisition'),
          cc.seq(cc.trial('A-'), repeat=n_extinction_few, name='extinction'),
          cc.seq(cc.trial('A+'), repeat=n_reacquisition_few, name='reacquisition'),
        ),
      'control many':
        cc.seq(
          cc.seq(cc.trial('-'), repeat=n_acquisition, name='acquisition'),
          cc.seq(cc.trial('-'), repeat=n_extinction_many, name='extinction'),
          cc.seq(cc.trial('A+'), repeat=n_reacquisition_many, name='reacquisition'),
        ),
      'extinction many':
        cc.seq(
          cc.seq(cc.trial('A+'), repeat=n_acquisition, name='acquisition'),
          cc.seq(cc.trial('A-'), repeat=n_extinction_many, name='extinction'),
          cc.seq(cc.trial('A+'), repeat=n_reacquisition_many, name='reacquisition'),
        ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='elevation score',
      citation='Ricker & Bouton (1996)',
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
          ['control few', 1, 1.4],
          ['control few', 2, 3.4],
          ['control few', 3, 2.6],
          ['control few', 4, 4.6],
          ['control few', 5, 3.8],
          ['control few', 6, 6.2],
          ['control few', 7, 5.8],
          ['control few', 8, 7.6],
          ['extinction few', 1, 3.2],
          ['extinction few', 2, 6.0],
          ['extinction few', 3, 6.5],
          ['extinction few', 4, 7.4],
          ['extinction few', 5, 6.0],
          ['extinction few', 6, 6.4],
          ['extinction few', 7, 4.4],
          ['extinction few', 8, 5.4],
          ['control many', 1, 1.1],
          ['control many', 2, 3.1],
          ['control many', 3, 5.7],
          ['control many', 4, 5.8],
          ['control many', 5, 5.2],
          ['control many', 6, 6.5],
          ['control many', 7, 8.3],
          ['control many', 8, 8.8],
          ['control many', 9, 7.0],
          ['control many', 10, 7.8],
          ['control many', 11, 8.8],
          ['control many', 12, 7.9],
          ['extinction many', 1, 1.4],
          ['extinction many', 2, 4.5],
          ['extinction many', 3, 3.8],
          ['extinction many', 4, 4.7],
          ['extinction many', 5, 5.2],
          ['extinction many', 6, 5.4],
          ['extinction many', 7, 4.5],
          ['extinction many', 8, 5.3],
          ['extinction many', 9, 5.4],
          ['extinction many', 10, 6.2],
          ['extinction many', 11, 4.9],
          ['extinction many', 12, 4.3],
        ]
      ),
      id_vars=['group', 'session']
    )
    self.plots = [
      lambda df, ax, **kwargs: cc.plot_lines(
        df, ax=ax, x='session', xlabel=kwargs['xlabel'], ylabel=kwargs['ylabel'], legend_cols=2
      )
    ]

  def simulated_results(self):
    return cc.trials_to_sessions(
      pd.melt(
        self.dataframe(
          lambda x: {
            'A': cc.conditioned_response(x['timesteps'], x['responses'], ['A']),
          } if x['phase'] == 'reacquisition' else None,
          include_trial=False,
          include_trial_in_phase=True,
        ),
        id_vars=['group', 'trial in phase']
      ).groupby(['group', 'trial in phase', 'variable'], sort=False).mean().reset_index(),
      1,
      trial_name='trial in phase',
    )
