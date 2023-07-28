name = "neuralplayground"
import os
import glob


dir_files = glob.glob("./*")
for files in dir_files:
    if files == "./config_params.yaml":
        CONFIG_PATH = files
    else:
        CONFIG_PATH = None

from .config import generate_config_file
