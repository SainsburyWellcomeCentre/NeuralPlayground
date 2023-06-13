import os
import shutil

import yaml

import neuralplayground

from .plot_config import PlotsConfig


def _load_config(config_path: str = None):
    """Load config from a given path

    Parameters
    ----------
    config_path: str
        Path to the config file

    Returns
    -------
    config: NPGConfig
        Config object
    """

    with open(config_path, "r") as file:
        CONFIG_FILE_YAML = yaml.safe_load(file)
        print(config_path)
    PLOT_CONFIG = PlotsConfig(CONFIG_FILE_YAML["plot_config"])
    return PLOT_CONFIG


def generate_config_file():
    """Generate a config file in the current directory

    Parameters
    ----------
    config_path: str
        Path to the config file

    Returns
    -------
    config: NPGConfig
        Config object
    """

    shutil.copyfile(os.path.join(neuralplayground.__path__[0], "config", "default_config.yaml"), "./config_params.yaml")
