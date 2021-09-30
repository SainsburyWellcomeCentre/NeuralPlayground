import pandas as pd
import ccnlab.benchmarks.classical.core as cc


@cc.registry.register
class Discrimination_ReinforcedVsNonreinforced(cc.ClassicalConditioningExperiment):
  """A reinforced CS elicits significantly greater CR than a non-reinforced CS.

  Source: 4.1 - Figure 10
  """
  def __init__(self, n=250):
    super().__init__({
      'main':
        cc.seq(
          cc.seq(cc.trial('A+'), name='train-A'),
          cc.seq(cc.trial('B-'), name='train-B'),
          repeat=n,
        ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='conditioned response [%]',
      citation='Campolattaro et al. (2008)',
      animal='rat',
      cs='visual, auditory',
      us='shock',
      response='eyeblink',
      preparation='eyeblink conditioning',
    )
    self.empirical_results = pd.melt(
      pd.DataFrame(
        columns=['group', 'session', 'A', 'B'],
        data=[
          ['main', 1, 20, 31],
          ['main', 2, 19, 39],
          ['main', 3, 20, 48],
          ['main', 4, 21, 65],
          ['main', 5, 12, 78],
          ['main', 6, 13, 80],
          ['main', 7, 12, 81],
          ['main', 8, 13, 86],
          ['main', 9, 13, 81],
          ['main', 10, 13, 85],
        ]
      ),
      id_vars=['group', 'session']
    )
    self.plots = [
      lambda df, ax, **kwargs: cc.plot_lines(
        df, ax=ax, x='session', xlabel=kwargs['xlabel'], ylabel=kwargs['ylabel'], legend=False
      )
    ]

  def simulated_results(self):
    return cc.trials_to_sessions(
      pd.melt(
        self.dataframe(
          lambda x: {
            'A': cc.conditioned_response(x['timesteps'], x['responses'], ['A']),
          } if x['phase'] == 'train-A' else {
            'B': cc.conditioned_response(x['timesteps'], x['responses'], ['B']),
          } if x['phase'] == 'train-B' else None,
          include_trial=False,
          include_trial_in_phase=True,
        ),
        id_vars=['group', 'trial in phase']
      ).groupby(['group', 'trial in phase', 'variable'], sort=False).mean().reset_index(),
      25,
      trial_name='trial in phase'
    )


@cc.registry.register
class Discrimination_PositivePatterning(cc.ClassicalConditioningExperiment):
  """Reinforced AB+ intermixed with non-reinforced A- and B- results in responding to AB that is
  stronger than the sum of the individual responses to A and B.

  Source: 4.2 - Figure 11
  """
  def __init__(self, n=480):
    super().__init__({
      'main':
        cc.seq(
          cc.seq(cc.trial('A-'), name='train-A'),
          cc.seq(cc.trial('B-'), name='train-B'),
          cc.seq(cc.trial('AB+'), name='train-AB'),
          repeat=n,
        ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='conditioned response [%]',
      citation='Bellingham et al. (1985)',
      animal='rat',
      cs='visual, auditory',
      us='water',
      response='drinking',
      preparation='appetitive conditioning',
    )
    self.empirical_results = pd.melt(
      pd.DataFrame(
        columns=['group', 'session', 'A', 'B', 'AB'],
        data=[
          ['main', 1, 11, 23, 21],
          ['main', 2, 41, 51, 53],
          ['main', 3, 61, 73, 98],
          ['main', 4, 54, 56, 93],
          ['main', 5, 63, 61, 93],
          ['main', 6, 46, 48, 93],
          ['main', 7, 36, 41, 86],
          ['main', 8, 38, 30, 85],
          ['main', 9, 28, 30, 85],
          ['main', 10, 31, 33, 83],
          ['main', 11, 31, 23, 84],
          ['main', 12, 31, 29, 85],
          ['main', 13, 28, 21, 88],
          ['main', 14, 24, 21, 88],
          ['main', 15, 24, 13, 80],
          ['main', 16, 29, 17, 88],
          ['main', 17, 24, 16, 80],
          ['main', 18, 21, 18, 80],
          ['main', 19, 26, 21, 84],
          ['main', 20, 26, 26, 87],
          ['main', 21, 24, 16, 84],
          ['main', 22, 25, 24, 91],
          ['main', 23, 16, 21, 86],
          ['main', 24, 21, 24, 83],
        ]
      ),
      id_vars=['group', 'session']
    )
    self.plots = [
      lambda df, ax, **kwargs: cc.plot_lines(
        df, ax=ax, x='session', xlabel=kwargs['xlabel'], ylabel=kwargs['ylabel'], legend=False
      )
    ]

  def simulated_results(self):
    return cc.trials_to_sessions(
      pd.melt(
        self.dataframe(
          lambda x: {
            'A': cc.conditioned_response(x['timesteps'], x['responses'], ['A']),
          } if x['phase'] == 'train-A' else {
            'B': cc.conditioned_response(x['timesteps'], x['responses'], ['B']),
          } if x['phase'] == 'train-B' else {
            'AB': cc.conditioned_response(x['timesteps'], x['responses'], ['A', 'B']),
          } if x['phase'] == 'train-AB' else None,
          include_trial=False,
          include_trial_in_phase=True,
        ),
        id_vars=['group', 'trial in phase']
      ).groupby(['group', 'trial in phase', 'variable'], sort=False).mean().reset_index(),
      20,
      trial_name='trial in phase'
    )


@cc.registry.register
class Discrimination_NegativePatterning(cc.ClassicalConditioningExperiment):
  """Non-reinforced AB- intermixed with reinforced A+ and B+ results in responding to AB that is
  weaker than the sum of the individual responses to A and B.

  Source: 4.3 - Figure 12
  """
  def __init__(self, n=480):
    super().__init__({
      'main':
        cc.seq(
          cc.seq(cc.trial('A+'), name='train-A'),
          cc.seq(cc.trial('B+'), name='train-B'),
          cc.seq(cc.trial('AB-'), name='train-AB'),
          repeat=n,
        ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='conditioned response [%]',
      citation='Bellingham et al. (1985)',
      animal='rat',
      cs='visual, auditory',
      us='water',
      response='drinking',
      preparation='appetitive conditioning',
    )
    self.empirical_results = pd.melt(
      pd.DataFrame(
        columns=['group', 'session', 'A', 'B', 'AB'],
        data=[
          ['main', 1, 38, 20, 26],
          ['main', 2, 51, 40, 81],
          ['main', 3, 78, 73, 93],
          ['main', 4, 81, 80, 93],
          ['main', 5, 89, 85, 93],
          ['main', 6, 88, 85, 83],
          ['main', 7, 88, 92, 73],
          ['main', 8, 78, 83, 66],
          ['main', 9, 88, 88, 58],
          ['main', 10, 91, 83, 51],
          ['main', 11, 85, 83, 53],
          ['main', 12, 80, 91, 48],
          ['main', 13, 96, 93, 48],
          ['main', 14, 96, 88, 36],
          ['main', 15, 96, 88, 36],
          ['main', 16, 88, 83, 36],
          ['main', 17, 86, 88, 27],
          ['main', 18, 80, 92, 23],
          ['main', 19, 88, 88, 28],
          ['main', 20, 88, 88, 26],
          ['main', 21, 80, 88, 28],
          ['main', 22, 84, 85, 26],
          ['main', 23, 80, 88, 31],
          ['main', 24, 73, 86, 26],
        ]
      ),
      id_vars=['group', 'session']
    )
    self.plots = [
      lambda df, ax, **kwargs: cc.plot_lines(
        df, ax=ax, x='session', xlabel=kwargs['xlabel'], ylabel=kwargs['ylabel'], legend=False
      )
    ]

  def simulated_results(self):
    return cc.trials_to_sessions(
      pd.melt(
        self.dataframe(
          lambda x: {
            'A': cc.conditioned_response(x['timesteps'], x['responses'], ['A']),
          } if x['phase'] == 'train-A' else {
            'B': cc.conditioned_response(x['timesteps'], x['responses'], ['B']),
          } if x['phase'] == 'train-B' else {
            'AB': cc.conditioned_response(x['timesteps'], x['responses'], ['A', 'B']),
          } if x['phase'] == 'train-AB' else None,
          include_trial=False,
          include_trial_in_phase=True,
        ),
        id_vars=['group', 'trial in phase']
      ).groupby(['group', 'trial in phase', 'variable'], sort=False).mean().reset_index(),
      20,
      trial_name='trial in phase'
    )


@cc.registry.register
class Discrimination_NegativePatterningCommonCue(cc.ClassicalConditioningExperiment):
  """Adding a common cue C to negative patterning decreases discrimination.

  Source: 4.5 - Figure 14
  """
  def __init__(self, n=70):
    super().__init__({
      'control':
        cc.seq(
          cc.seq(cc.trial('A+'), name='train-A'),
          cc.seq(cc.trial('B+'), name='train-B'),
          cc.seq(cc.trial('AB-'), name='train-AB'),
          repeat=n,
        ),
      'common cue':
        cc.seq(
          cc.seq(cc.trial('AC+'), name='train-AC'),
          cc.seq(cc.trial('BC+'), name='train-BC'),
          cc.seq(cc.trial('ABC-'), name='train-ABC'),
          repeat=n,
        ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='conditioned response [#/min]',
      citation='Redhead & Pearce (1998)',
      animal='pigeon',
      cs='visual',
      us='food',
      response='feeding',
      preparation='appetitive conditioning',
    )
    self.empirical_results = pd.concat([
      pd.melt(
        pd.DataFrame(
          columns=['group', 'session', 'A', 'B', 'AB'],
          data=[
            ['control', 1, 68, 56, 36],
            ['control', 2, 72, 60, 26],
            ['control', 3, 75, 64, 15],
            ['control', 4, 78, 60, 2],
            ['control', 5, 78, 74, 2],
            ['control', 6, 86, 70, 1],
            ['control', 7, 84, 76, 0],
          ]
        ),
        id_vars=['group', 'session']
      ),
      pd.melt(
        pd.DataFrame(
          columns=['group', 'session', 'AC', 'BC', 'ABC'],
          data=[
            ['common cue', 1, 64, 64, 64],
            ['common cue', 2, 64, 64, 64],
            ['common cue', 3, 74, 80, 64],
            ['common cue', 4, 68, 75, 36],
            ['common cue', 5, 82, 82, 28],
            ['common cue', 6, 86, 90, 32],
            ['common cue', 7, 92, 88, 18],
          ]
        ),
        id_vars=['group', 'session']
      )
    ])

    self.plots = [
      lambda df, ax, **kwargs: cc.plot_lines(
        df, ax=ax, x='session', xlabel=kwargs['xlabel'], ylabel=kwargs['ylabel'], legend=False
      )
    ]

  def simulated_results(self):
    return cc.trials_to_sessions(
      pd.melt(
        self.dataframe(
          lambda x: {
            'A': cc.conditioned_response(x['timesteps'], x['responses'], ['A']),
          } if x['phase'] == 'train-A' else {
            'B': cc.conditioned_response(x['timesteps'], x['responses'], ['B']),
          } if x['phase'] == 'train-B' else {
            'AB': cc.conditioned_response(x['timesteps'], x['responses'], ['A', 'B']),
          } if x['phase'] == 'train-AB' else {
            'AC': cc.conditioned_response(x['timesteps'], x['responses'], ['A', 'C']),
          } if x['phase'] == 'train-AC' else {
            'BC': cc.conditioned_response(x['timesteps'], x['responses'], ['B', 'C']),
          } if x['phase'] == 'train-BC' else {
            'ABC': cc.conditioned_response(x['timesteps'], x['responses'], ['A', 'B', 'C']),
          } if x['phase'] == 'train-ABC' else None,
          include_trial=False,
          include_trial_in_phase=True,
        ),
        id_vars=['group', 'trial in phase']
      ).dropna().groupby(['group', 'trial in phase', 'variable'], sort=False).mean().reset_index(),
      10,
      trial_name='trial in phase'
    )


@cc.registry.register
class Discrimination_NegativePatterningThreeCues(cc.ClassicalConditioningExperiment):
  """Non-reinforced ABC- intermixed with reinforced A+ and BC+ results in responding to ABC that is
  weaker than the sum of the individual responses to A and BC.

  Source: 4.6 - Figure 15
  """
  def __init__(self, n=50):
    super().__init__({
      'main':
        cc.seq(
          cc.seq(cc.trial('A+'), name='train-A'),
          cc.seq(cc.trial('BC+'), name='train-BC'),
          cc.seq(cc.trial('ABC-'), name='train-ABC'),
          repeat=n,
        ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='conditioned response [#/min]',
      citation='Redhead & Pearce (1995)',
      animal='pigeon',
      cs='visual',
      us='food',
      response='feeding',
      preparation='appetitive conditioning',
    )
    self.empirical_results = pd.melt(
      pd.DataFrame(
        columns=['group', 'session', 'A', 'BC', 'ABC'],
        data=[
          ['main', 1, 80, 55, 64],
          ['main', 2, 142, 93, 101],
          ['main', 3, 158, 110, 95],
          ['main', 4, 164, 115, 75],
          ['main', 5, 168, 135, 72],
          ['main', 6, 160, 120, 53],
          ['main', 7, 164, 143, 53],
          ['main', 8, 160, 141, 43],
          ['main', 9, 166, 152, 53],
          ['main', 10, 169, 158, 51],
        ]
      ),
      id_vars=['group', 'session']
    )
    self.plots = [
      lambda df, ax, **kwargs: cc.plot_lines(
        df, ax=ax, x='session', xlabel=kwargs['xlabel'], ylabel=kwargs['ylabel'], legend=False
      )
    ]

  def simulated_results(self):
    return cc.trials_to_sessions(
      pd.melt(
        self.dataframe(
          lambda x: {
            'A': cc.conditioned_response(x['timesteps'], x['responses'], ['A']),
          } if x['phase'] == 'train-A' else {
            'BC': cc.conditioned_response(x['timesteps'], x['responses'], ['B', 'C']),
          } if x['phase'] == 'train-BC' else {
            'ABC': cc.conditioned_response(x['timesteps'], x['responses'], ['A', 'B', 'C']),
          } if x['phase'] == 'train-ABC' else None,
          include_trial=False,
          include_trial_in_phase=True,
        ),
        id_vars=['group', 'trial in phase']
      ).groupby(['group', 'trial in phase', 'variable'], sort=False).mean().reset_index(),
      5,
      trial_name='trial in phase'
    )


@cc.registry.register
class Discrimination_Biconditional(cc.ClassicalConditioningExperiment):
  """Biconditional discrimination between compounds (AC+/BD+ vs. AD-/BC-, where no single CS
  predicts reinforcement or non-reinforcement) is possible but harder than component discrimination
  between compounds (AC+/AD+ vs. BC-/BD-, where A and B predict reinforcement and non-reinforcement,
  respectively).

  Source: 4.7, 4.9 - Figure 16
  """
  def __init__(self, n_componennt=50, n_biconditional=100):
    super().__init__({
      'component':
        cc.seq(
          cc.seq(cc.trial('AC+'), name='train-AC'),
          cc.seq(cc.trial('AD+'), name='train-AD'),
          cc.seq(cc.trial('BC-'), name='train-BC'),
          cc.seq(cc.trial('BD-'), name='train-BD'),
          repeat=n_componennt,
        ),
      'biconditional':
        cc.seq(
          cc.seq(cc.trial('AC+'), name='train-AC'),
          cc.seq(cc.trial('AD-'), name='train-AD'),
          cc.seq(cc.trial('BC-'), name='train-BC'),
          cc.seq(cc.trial('BD+'), name='train-BD'),
          repeat=n_biconditional,
        ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='conditioned response [%]',
      citation='Saavedra (1975)',
      animal='rabbit',
      cs='visual, auditory',
      us='shock',
      response='eyeblink',
      preparation='eyeblink conditioning',
    )
    self.empirical_results = pd.melt(
      pd.DataFrame(
        columns=['group', 'session', '+', '\u2212'],  # Use "minus" character for prettier plot.
        data=[
          ['component', 1, 34, 30],
          ['component', 2, 70, 39],
          ['component', 3, 90, 42],
          ['component', 4, 91, 33],
          ['component', 5, 91, 34],
          ['biconditional', 1, 48, 37],
          ['biconditional', 2, 65, 64],
          ['biconditional', 3, 85, 64],
          ['biconditional', 4, 88, 72],
          ['biconditional', 5, 86, 62],
          ['biconditional', 6, 94, 63],
          ['biconditional', 7, 91, 53],
          ['biconditional', 8, 88, 47],
          ['biconditional', 9, 94, 62],
          ['biconditional', 10, 93, 50],
        ]
      ),
      id_vars=['group', 'session']
    )
    self.plots = [
      lambda df, ax, **kwargs: cc.plot_lines(
        df, ax=ax, x='session', xlabel=kwargs['xlabel'], ylabel=kwargs['ylabel'], legend=False
      )
    ]

  def simulated_results(self):
    return cc.trials_to_sessions(
      pd.melt(
        self.dataframe(
          lambda x: {
            '+': cc.conditioned_response(x['timesteps'], x['responses'], ['A', 'C']),
          } if x['phase'] == 'train-AC' else {
            ('+' if x['group'] == 'component' else '\u2212'):
              cc.conditioned_response(x['timesteps'], x['responses'], ['A', 'D']),
          } if x['phase'] == 'train-AD' else {
            '\u2212': cc.conditioned_response(x['timesteps'], x['responses'], ['B', 'C']),
          } if x['phase'] == 'train-BC' else {
            ('\u2212' if x['group'] == 'component' else '+'):
              cc.conditioned_response(x['timesteps'], x['responses'], ['B', 'D']),
          } if x['phase'] == 'train-BD' else None,
          include_trial=False,
          include_trial_in_phase=True,
        ),
        id_vars=['group', 'trial in phase']
      ).groupby(['group', 'trial in phase', 'variable'], sort=False).mean().reset_index(),
      10,
      trial_name='trial in phase'
    )


@cc.registry.register
class Discrimination_FeaturePositive(cc.ClassicalConditioningExperiment):
  """Reinforced BA+, alternated with non-reinforced A-, results in stronger responding to BA than A
  alone. In the simultaneous case (BA+), B gains an excitatory association with the US; in the
  serial case (B -> A+), B does not gain an excitatory association with the US.

  Source: 4.12, 4.13 - Figure 20
  """
  def __init__(self, n_sim=360, n_serial=780):
    super().__init__({
      'simultaneous':
        cc.seq(
          cc.seq(cc.trial('BA+'), name='train-BA'),
          cc.seq(cc.trial('A-'), name='train-A'),
          repeat=n_sim // 2,
        ),
      'serial':
        cc.seq(
          cc.seq(cc.trial(('B', 0, 4), ('A', 4, 8), ('+', 7, 8)), name='train-BA'),
          cc.seq(cc.trial('A-'), name='train-A'),
          repeat=n_serial // 2,
        ),
    })
    self.meta = dict(
      ylabel='conditioned response',
      ydetail='conditioned response [%]',
      citation='Ross & Holland (1981)',
      animal='rat',
      cs='visual, auditory',
      us='food',
      response='head jerk',
      preparation='appetitive conditioning',
    )
    self.empirical_results = pd.melt(
      pd.DataFrame(
        columns=['group', 'session', 'BA', 'A'],
        data=[
          ['simultaneous', 1, 2, 3],
          ['simultaneous', 2, 4, 2],
          ['simultaneous', 3, 21, 1],
          ['simultaneous', 4, 56, 2],
          ['simultaneous', 5, 54, 0],
          ['simultaneous', 6, 65, 0],
          ['serial', 1, 0, 0],
          ['serial', 2, 4, 1],
          ['serial', 3, 0, 4],
          ['serial', 4, 21, 21],
          ['serial', 5, 39, 25],
          ['serial', 6, 51, 30],
          ['serial', 7, 51, 24],
          ['serial', 8, 57, 13],
          ['serial', 9, 36, 11],
          ['serial', 10, 49, 11],
          ['serial', 11, 36, 7],
          ['serial', 12, 57, 9],
          ['serial', 13, 53, 9],
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
          } if x['phase'] == 'train-A' else {
            # Only measure during A (i.e. tone in original experiment).
            'BA': cc.conditioned_response(x['timesteps'], x['responses'], ['A']),
          } if x['phase'] == 'train-BA' else None,
          include_trial=False,
          include_trial_in_phase=True,
        ),
        id_vars=['group', 'trial in phase']
      ).groupby(['group', 'trial in phase', 'variable'], sort=False).mean().reset_index(),
      30,
      trial_name='trial in phase'
    )


@cc.registry.register
class Discrimination_FeatureNegative(cc.ClassicalConditioningExperiment):
  """Non-reinforced BA-, alternated with reinforced A+, results in weaker responding to BA than A
  alone. In the simultaneous case (BA-), B gains an inhibitory association with the US; in the
  serial case (B -> A-), B does not gain an inhibitory association with the US.

  Source: 4.14, 4.15 - Figure 21
  """
  def __init__(self, n_acquisition=2, n_sim=3, n_serial=12, tps=4):
    super().__init__({
      'simultaneous':
        cc.seq(
          cc.seq(cc.trial('A+'), repeat=n_acquisition * tps, name='acquisition'),
          cc.seq(
            cc.seq(cc.trial('BA-'), name='train-BA'),
            cc.seq(cc.trial('A+'), name='train-A'),
            repeat=n_sim * tps,
          ),
        ),
      'serial':
        cc.seq(
          cc.seq(cc.trial('A+'), repeat=n_acquisition * tps, name='acquisition'),
          cc.seq(
            cc.seq(cc.trial(('B', 0, 4), ('A', 4, 8)), name='train-BA'),
            cc.seq(cc.trial('A+'), name='train-A'),
            repeat=n_serial * tps,
          ),
        )
    })
    self._tps = tps
    self.meta = dict(
      ylabel='suppression ratio',
      ydetail='suppression ratio',
      citation='Holland (1984)',
      animal='rat',
      cs='visual, auditory',
      us='shock',
      response='bar pressing',
      preparation='fear conditioning',
    )
    self.empirical_results = pd.melt(
      pd.DataFrame(
        columns=['group', 'session', 'BA', 'A'],
        data=[
          ['simultaneous', 1, 0.34, 0.17],
          ['simultaneous', 2, 0.39, 0.07],
          ['simultaneous', 3, 0.49, 0.06],
          ['serial', 1, 0.15, 0.13],
          ['serial', 2, 0.19, 0.16],
          ['serial', 3, 0.23, 0.16],
          ['serial', 4, 0.18, 0.07],
          ['serial', 5, 0.22, 0.15],
          ['serial', 6, 0.29, 0.08],
          ['serial', 7, 0.22, 0.10],
          ['serial', 8, 0.29, 0.10],
          ['serial', 9, 0.33, 0.13],
          ['serial', 10, 0.40, 0.14],
          ['serial', 11, 0.41, 0.08],
          ['serial', 12, 0.43, 0.14],
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
            'A': cc.suppression_ratio(x['timesteps'], x['responses'], ['A']),
          } if x['phase'] == 'train-A' else {
            # Only measure during A (i.e. noise in original experiment).
            'BA': cc.suppression_ratio(x['timesteps'], x['responses'], ['A']),
          } if x['phase'] == 'train-BA' else None,
          include_trial=False,
          include_trial_in_phase=True,
        ),
        id_vars=['group', 'trial in phase']
      ).groupby(['group', 'trial in phase', 'variable'], sort=False).mean().reset_index(),
      self._tps,
      trial_name='trial in phase'
    )
