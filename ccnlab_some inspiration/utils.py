import fnmatch
from collections import defaultdict


def listdict_to_dictlist(ld):
  dl = defaultdict(list)
  keys = set(k for d in ld for k in d)
  for d in ld:
    for k in keys:
      dl[k].append(d[k] if k in d else None)
  return dl


class SearchableRegistry:
  def __init__(self):
    self.items = {}

  def __call__(self, *args, **kwargs):
    return self.get(*args, **kwargs)

  def register(self, item):
    self.items[item.__name__] = item

  def get(self, *globs, call=True):
    """Return items that match any of the specified globs. By default, assumes that items are 
    callable (i.e. a function or constructor that requires no arguments) and calls them."""
    names = self.items.keys()
    if len(globs) > 0:
      names = [name for name in names if any(fnmatch.fnmatch(name, glob) for glob in globs)]
    return [self.items[name]() if call else self.items[name] for name in names]
