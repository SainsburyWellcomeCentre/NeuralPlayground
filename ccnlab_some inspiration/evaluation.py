import pandas as pd
from IPython import embed
import numpy as np


def _dfs_to_values(df1, df2, value_name='value'):
  # Match corresponding rows based on all column values except the value column,
  # ignoring rows that don't have a match.
  merged = df1.merge(df2, on=[x for x in df1.columns if x != value_name])
  values1 = merged['{}_x'.format(value_name)]
  values2 = merged['{}_y'.format(value_name)]
  return values1, values2


def correlation(df1, df2, value_name='value'):
  values1, values2 = _dfs_to_values(df1, df2, value_name)
  r = values1.corr(values2, method='pearson')
  return r


def ratio_of_ratios(df1, df2, value_name='value'):
  values1, values2 = _dfs_to_values(df1, df2, value_name)
  values1 = list(values1)
  values2 = list(values2)
  assert len(values1) == 2
  assert len(values2) == 2
  r1 = values1[1] / values1[0]
  r2 = values2[1] / values1[0]
  if r1 > r2:
    r1, r2 = r2, r1
  return r1 / r2
