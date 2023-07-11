from .plot_config import PlotsConfig
import yaml
import os
import neuralplayground
from .load_config import _load_config, _get_state_labels
from neuralplayground import CONFIG_PATH

# Load config file
if CONFIG_PATH is None:
    CONFIG_PATH = os.path.join(neuralplayground.__path__[0], "config", "default_config.yaml")
    PLOT_CONFIG = _load_config(CONFIG_PATH)
else:
    PLOT_CONFIG = _load_config(CONFIG_PATH)

from .load_config import generate_config_file

STATE_LABELS = _get_state_labels()
