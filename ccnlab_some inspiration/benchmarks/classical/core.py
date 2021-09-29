import random
import textwrap
from collections import namedtuple, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

from ccnlab.utils import listdict_to_dictlist, SearchableRegistry

# ==================================================================================================
# Syntax Tree Definitions

Stimulus = namedtuple('Stimulus', ['stim', 'start', 'end', 'mag'], defaults=(1,))

# Trial(cs=(Stimulus('A', 4, 8, 1),), ctx='K1', us=Stimulus('+', 7, 8, 1) or None)
Trial = namedtuple('Trial', ['cs', 'ctx', 'us'])

# Sample(prob={ x: 1, y: 1 })
Sample = namedtuple('Sample', ['prob'])

# Sequence(children=(...), repeat=1, name='train' or None)
Sequence = namedtuple('Sequence', ['children', 'repeat', 'name'], defaults=(1, None))


def trial(*args, ctx='K1', us_mapping={'-': 0, '+': 1, '#': 2}, cs_active=(4, 8), us_active=(7, 8)):
  cs = tuple()
  us = None
  assert isinstance(ctx, str)
  if isinstance(args[0], str):
    # Usage: trial('AB+')
    assert len(args) == 1
    for char in args[0]:
      if char in us_mapping:
        us = Stimulus(char, us_active[0], us_active[1], us_mapping[char])
      else:
        cs += (Stimulus(char, cs_active[0], cs_active[1], 1),)
  else:
    for item in args:
      if len(item) == 4:
        # Usage: trial(('A', 4, 8, 0.5), ('B', 4, 8, 0.5), ('+', 7, 8, 0.8))
        stim, start, end, mag = item
        assert isinstance(stim, str)
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert isinstance(mag, (int, float))
        us = Stimulus(stim, start, end, mag)
      else:
        # Usage: trial(('A', 4, 8), ('B', 4, 8), ('+', 7, 8))
        assert len(item) == 3
        stim, start, end = item
        if stim in us_mapping:
          us = Stimulus(stim, start, end, us_mapping[stim])
        else:
          cs += (Stimulus(*item),)
  return Trial(cs, ctx=ctx, us=us)


def seq(*children, repeat=1, name=None):
  return Sequence(children, repeat=repeat, name=name)


def sample(prob):
  return Sample(prob)


# ==================================================================================================
# Syntax Tree Building


def _build_trial(node):
  cs, ctx, us = node
  length = 0

  cs_t = defaultdict(list)
  if cs is not None:
    for stim, start, end, mag in cs:
      length = max(length, end)
      for t in range(start, end):
        cs_t[t].append((stim, mag))

  us_t = defaultdict(int)
  if us is not None:
    stim, start, end, mag = us
    length = max(length, end)
    for t in range(start, end):
      us_t[t] = mag

  return [(cs_t[t], ctx, us_t[t]) for t in range(length)]


def build_stimuli(node):
  # (cs, ctx, us) = stimuli[trial][timestep]
  # phase = phases[trial]
  stimuli = []
  phases = []

  def recurse(node, phase):
    name = type(node).__name__
    if name == 'Trial':
      trial = _build_trial(node)
      stimuli.append(trial)
      phases.append(phase)
    if name == 'Sample':
      choices = list(node.prob.keys())
      weights = list(node.prob.values())
      choice = random.choices(choices, weights=weights)[0]
      recurse(choice, phase)
    if name == 'Sequence':
      for _ in range(node.repeat):
        for child in node.children:
          recurse(child, node.name if node.name is not None else phase)

  recurse(node, None)
  return stimuli, phases


def _repr_node(node):
  name = type(node).__name__
  if name == 'Trial':
    stimuli = []
    if node.cs is not None: stimuli.extend(node.cs)
    if node.us is not None: stimuli.append(node.us)
    if len(stimuli) == 0: return ''
    stimuli.sort(key=lambda x: x.start)
    prefix = stimuli[0].stim
    end = stimuli[0].end
    for curr in stimuli[1:]:
      if curr.start >= end:
        end = curr.end
        prefix += ' -> '
      else:
        end = max(end, curr.end)
      prefix += curr.stim
    return prefix
  if name == 'Sample':
    items = [
      '(prob={}) {{\n{}\n}}'.format(p, textwrap.indent(_repr_node(child), '  '))
      for child, p in node.prob.items()
    ]
    return '\n'.join(items)
  if name == 'Sequence':
    prefix = ''
    if node.name is not None: prefix += '{} '.format(node.name)
    if node.repeat > 1: prefix += '(repeat={}) '.format(node.repeat)
    inner = '\n'.join(textwrap.indent(_repr_node(child), '  ') for child in node.children)
    prefix += '{{\n{}\n}}'.format(inner)
    return prefix


def _repr_spec(spec):
  return '\n'.join(
    '{}:\n{}'.format(group, textwrap.indent(_repr_node(tree), '  ')) for group, tree in spec.items()
  )


# ==================================================================================================
# Experiment Base Class

registry = SearchableRegistry()


class ClassicalConditioningExperiment:
  """An `ClassicalConditioningExperiment` is an environment that simulates the schedule of stimuli presented in a real-world classical conditioning experiment drawn from peer-reviewed academic research, and facilitates the comparison of simulated results with empirical results. Developers
  should subclass this to implement a particular experiment.
  """
  def __init__(self, spec, response_key='response'):
    self.spec = spec
    self.response_key = response_key
    self.stimuli = {}
    self.phases = {}
    for group, node in spec.items():
      self.stimuli[group], self.phases[group] = build_stimuli(node)

    cs_space = set()
    ctx_space = set()
    for stimuli in self.stimuli.values():
      for trial in stimuli:
        for timestep in trial:
          cs, ctx, us = timestep
          cs_space.update(*[stim for stim, mag in cs])
          ctx_space.add(ctx)
    self.cs_space = tuple(sorted(cs_space))
    self.ctx_space = tuple(sorted(ctx_space))

    self.data = {}
    self.reset()

    self.name = self.__class__.__name__
    self.meta = {}
    self.empirical_results = None
    self.plots = []

  def reset(self):
    self.data = {
      group: [[defaultdict(list) for timestep in trial] for trial in stimuli]
      for group, stimuli in self.stimuli.items()
    }

  def stimulus(self, group, trial, timestep, vector=False):
    """Return the stimulus (tuple (cs, ctx, us))) of a timestep within a trial within a group.
    By default, `cs` is a list of (id str, magnitude float) pairs, `ctx` is an id str, and `us` is
    a magnitude float. Optionally, `cs` and `ctx` can instead be returned as a one-hot vector."""
    stimuli = self.stimuli[group][trial][timestep]
    if not vector: return stimuli
    cs, ctx, us = stimuli
    cs_vec = [next((mag for stim, mag in cs if x == stim), 0) for x in self.cs_space]
    ctx_vec = [1 if x == ctx else 0 for x in self.ctx_space]
    return cs_vec, ctx_vec, us

  def phase(self, group, trial):
    """Return the phase (str or None) of a trial within a group."""
    return self.phases[group][trial]

  def dataframe(
    self,
    fn,
    include_group=True,
    include_trial=True,
    include_phase=False,
    include_trial_in_phase=False
  ):
    """Apply the given processing function on the stimuli and responses to create rows in a
    dataframe."""
    # fn: (args) => { 'col': value, ... }
    # self.stimuli[group][trial]
    # self.data[group][trial][var][subject]
    data = []
    for g, group in self.stimuli.items():
      phase_counts = defaultdict(int)
      for i, timesteps in enumerate(group):
        p = self.phases[g][i]
        phase_counts[p] += 1
        # [timestep][var][subject] -> [var][timestep][subject] -> [timestep][subject]
        x = listdict_to_dictlist(self.data[g][i])[self.response_key]
        # [timestep][subject] -> [subject][timestep]
        x = zip(*x)
        for values in x:
          kwargs = {
            'responses': values,
            'timesteps': timesteps,
            'group': g,
            'trial': i + 1,
            'trial in phase': phase_counts[p],
            'phase': p,
          }
          custom_cols = fn(kwargs)
          if custom_cols is not None:
            info_cols = {}
            if include_group: info_cols['group'] = g
            if include_trial: info_cols['trial'] = i + 1
            if include_trial_in_phase: info_cols['trial in phase'] = phase_counts[p]
            if include_phase: info_cols['phase'] = p
            data.append({**info_cols, **custom_cols})
    return pd.DataFrame(data)

  def simulated_results(self):
    """Transform collected data into format identical to `self.empirical_results`."""
    raise NotImplementedError

  def schedule(self):
    """Return a string summarizing the schedule of stimuli presented in this experiment."""
    return _repr_spec(self.spec)

  def plot(self, show='both', figsize=(6, 4), axes=None, xlabel=True, ylabel=True):
    """Plot empirical and/or simulated results."""
    # show: 'empirical' | 'stimulated' | 'both'
    for plotfn in self.plots:
      if show == 'empirical':
        dfs = [self.empirical_results]
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]
        kind = ['empirical']
      elif show == 'stimulated':
        dfs = [self.simulated_results()]
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]
        kind = ['stimulated']
      else:
        dfs = [self.empirical_results, self.simulated_results()]
        fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))
        kind = ['empirical', 'stimulated']

      for i, df in enumerate(dfs):
        xlab = ''
        if xlabel:
          xlab = self.meta.get('xlabel', None)
          if kind[i] == 'empirical': xlab = self.meta.get('xdetail', xlab)
        ylab = ''
        if ylabel:
          ylab = self.meta.get('ylabel', None)
          if kind[i] == 'empirical': ylab = self.meta.get('ydetail', ylab)
        plotfn(df, axes[i], xlabel=xlab, ylabel=ylab, kind=kind[i])


# ==================================================================================================
# Data Processing


def sum_responses(stimuli, responses, during_cs=None, during_ctx=None, during_us=None):
  """Calculate sum of responses for a single trial, optionally only counting timesteps occuring
  during the specified `during_cs`, `during_ctx`, or `during_us` values. For `during_cs`, the `cs` must match all provided vlaue (will not count if is a subset)."""

  # (cs, ctx, us) = stimuli[timestep]
  # response = responses[timestep]
  def include(stimulus):
    cs, ctx, us = stimulus
    result = True
    if during_cs is not None:
      result &= len(cs) == len(during_cs) and all(stim in during_cs and mag > 0 for stim, mag in cs)
    if during_ctx is not None:
      result &= ctx == during_ctx
    if during_us is not None:
      result &= us > 0 if during_us else us == 0
    return result

  return sum(responses[i] for i, stimulus in enumerate(stimuli) if include(stimulus))


def conditioned_response(stimuli, responses, during_cs, during_us=False):
  """Calculates conditioned response for a single trial."""
  assert len(stimuli) == len(responses)
  num = sum_responses(stimuli, responses, during_cs=during_cs, during_us=during_us)
  denom = sum_responses(stimuli, [1] * len(responses), during_cs=during_cs, during_us=during_us)
  if denom == 0: return 0
  return float(num) / denom


def suppression_ratio(stimuli, responses, during_cs, during_us=False):
  """Calculates suppression ratio for a single trial."""
  assert len(stimuli) == len(responses)
  # Invert responses to mimic aversive behavior preparations (see paper for details).
  max_r = max(responses)
  responses = [max_r - r for r in responses]
  cs = sum_responses(stimuli, responses, during_cs=during_cs, during_us=during_us)
  ctx = sum_responses(stimuli, responses, during_cs=[], during_us=during_us)
  num = cs
  denom = cs + ctx
  if denom == 0: return 0
  return float(num) / denom


def trials_to_sessions(
  df,
  trials_per_session,
  keep_first=False,
  trial_name='trial',
  session_name='session',
  value_name='value',
):
  """Aggregate consecutive spans of trials into sessions through averaging. Spans have length
  `trials_per_session`. Optionally, the first trial can be included (via `keep_first`)."""
  first = df[df[trial_name] == 1].copy()
  first[trial_name] = 0
  first = first.rename(columns={trial_name: session_name})

  df = df.copy()
  df[trial_name] = np.ceil(df[trial_name].div(trials_per_session)).astype(int)
  df = df.rename(columns={trial_name: session_name})

  if keep_first: df = pd.concat((first, df))
  return df.groupby([col for col in df.columns if col != value_name],
                    sort=False).mean().reset_index()


# ==================================================================================================
# Plotting


def _line_labels(last, pos='value', sep=1, delta=0.1, max_iters=20):
  """Calculate y-position of line labels, minimizing label overlap."""
  labels = last.copy()
  for _ in range(max_iters):
    changed = False
    for i, row_i in labels.iterrows():
      for j, row_j in labels.iterrows():
        if j <= i: continue
        pos_i = row_i[pos]
        pos_j = row_j[pos]
        if abs(pos_i - pos_j) < sep:
          if pos_i < pos_j:
            labels.at[i, pos] -= delta
            labels.at[j, pos] += delta
          else:
            labels.at[i, pos] += delta
            labels.at[j, pos] -= delta
          changed = True
    if not changed: break
  return labels


def _pt_to_data_coord(ax, x, y):
  """Convert (x, y) value in points into (x', y') value in data coordinates on axes `ax`."""
  t = ax.transData.inverted()
  return t.transform((x, y)) - t.transform((0, 0))


def plot_lines(
  df,
  x='trial',
  y='value',
  group='group',
  split='variable',
  legend=True,
  label=True,
  xlabel=None,
  ylabel='',
  yaxis=None,
  ax=None,
  label_fontsize=12,
  legend_cols=1,
  legend_pos=(0.5, -0.3),
):
  """Plot line graph, coloring by `group` and splitting into multiple lines within a group using 
  `split`. Expects `df` in long-form.
  """
  if ax is None: fig, ax = plt.subplots()
  palette = sns.color_palette()
  g = sns.lineplot(data=df, x=x, y=y, hue=group, units=split, estimator=None, markers=True, ax=ax)
  sns.despine()
  if legend:
    ax.legend(loc='best', ncol=legend_cols)  #, bbox_to_anchor=legend_pos)
  elif ax.get_legend() is not None:
    ax.get_legend().remove()
  if yaxis is not None:
    g.set(ylim=(yaxis[0], yaxis[1]), yticks=np.arange(yaxis[0], yaxis[1] + yaxis[2], step=yaxis[2]))
  ax.xaxis.set_major_locator(MaxNLocator(integer=True))
  ax.set_xlabel(x)
  if xlabel is not None: ax.set_xlabel(xlabel)
  if ylabel is not None: ax.set_ylabel(ylabel)

  if label is not None:
    groups = list(df[group].unique())
    last = df.sort_values([x]).groupby([group, split]).last().reset_index()
    sep = _pt_to_data_coord(ax, 0, label_fontsize)[1]
    labels = _line_labels(last, pos=y, sep=sep, delta=sep / 10)
    for i, row in labels.iterrows():
      color = palette[groups.index(row[group]) % len(palette)]
      ax.annotate(
        row[split], (row[x], row[y]),
        xytext=(4, 0),
        textcoords='offset points',
        horizontalalignment='left',
        verticalalignment='center',
        color=color,
        linespacing=1,
        fontsize=label_fontsize,
        fontweight='bold'
      )


def plot_bars(
  df,
  x='group',
  y='value',
  split='variable',
  label='variable',
  xlabel=None,
  ylabel='',
  yaxis=None,
  ax=None,
  barfrac=0.6,
  label_fontsize=12,
  wrap=None,
):
  """Plot bar graph, coloring by `group` and splitting into multiple bars within a group using 
  `split`. Expects `df` in long-form.
  """
  if ax is None: fig, ax = plt.subplots()
  xs = df[x].unique()
  splits = df[split].unique() if split in df else [None]
  xvals = pd.Series(np.arange(len(xs)), index=xs)  # Leftmost x-coord of bars.
  width = min(barfrac, 0.9 / len(splits))  # Use preferred bar width unless not enough space.
  palette = sns.color_palette()
  colors = pd.Series([palette[x % len(palette)] for x in range(len(xs))], index=xs)

  for i, s in enumerate(splits):
    df_split = df[df[split] == s] if split in df else df
    # Offset by i * width to space splits; offset by width / 2 because bar() uses middle x-coord.
    bar = ax.bar(
      xvals[df_split[x]] + i * width + width / 2, df_split[y], width, color=colors[df_split[x]]
    )
    if label is not None:
      labels = ax.bar_label(
        bar,
        labels=df_split[label],
        padding=2,
        fontsize=label_fontsize,
        fontweight='bold',
        linespacing=1
      )
      for i, color in enumerate(colors[df_split[x]]):
        labels[i].set_color(color)

  sns.despine()
  ax.set_xticks(xvals + len(splits) * width / 2)  # Middle x-coord of split group.
  ax.set_xticklabels([
    '\n'.join(textwrap.wrap(x, wrap, break_long_words=False)) if wrap is not None else x for x in xs
  ])
  # ax.margins(x=0)
  # ax.set_xlim(len(splits) * width / 2 - 1, len(xs) + len(splits) * width / 2)
  if yaxis is not None:
    ax.set_ylim(yaxis[0], yaxis[1])
    ax.set_yticks(np.arange(yaxis[0], yaxis[1] + yaxis[2], step=yaxis[2]))
  ax.set_xlabel(x)
  if xlabel is not None: ax.set_xlabel(xlabel)
  if ylabel is not None: ax.set_ylabel(ylabel)
