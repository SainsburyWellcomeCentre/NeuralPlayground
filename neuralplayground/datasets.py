"""Module for fetching and loading datasets.

This module provides functions for fetching and loading data used in tests,
examples, and tutorials. The data are stored in a remote repository on GIN
and are downloaded to the user's local machine the first time they are used.
"""

from pathlib import Path

import pooch

# URL to GIN data repository where the experimental data are hosted
DATA_URL = "https://gin.g-node.org/SainsburyWellcomeCentre/NeuralPlayground/raw/master"

# Data to be downloaded and cached in ~/.NeuralPlayground/data
LOCAL_DATA_DIR = Path("~", ".NeuralPlayground", "data").expanduser()
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

# A pooch data registry object
# Datasets are in the "data" subfolder as zip files - format: {dataset_name}.zip
DATASET_REGISTRY = pooch.create(
    path=LOCAL_DATA_DIR,
    base_url=f"{DATA_URL}/data/",
    registry={
        "hafting_2008.zip": "18934257966c8017e0d86909576468fc7fef5cf5388042b89ffa0833aeb12f04",  # noqa: E501
        "sargolini_2006.zip": "ca5011e32bb510491e81d2e1d74c45b4ffd1e5c3c5f326237fadd9b2a8867bc3",  # noqa: E501
        "wernle_2018.zip": "eed1ee8fda8f0ea12e39323db9fecc3e8bb61d3e18aac7dd88ec32d402e5982e",  # noqa: E501
    },
)

dataset_names = [n.split(".")[0] for n in DATASET_REGISTRY.registry.keys()]


def fetch_data_path(
    dataset_name: str,
    progressbar: bool = True,
):
    """Download and cache a dataset from the GIN repository.

    Parameters
    ----------
    dataset_name : str
        The name of one the available datasets, e.g. "hafting_2008".
    progressbar : bool
        If True, show a progress bar while downloading the data.
        Defaults to True.

    Returns
    -------
    str
        Path to the downloaded dataset
    """
    if dataset_name not in dataset_names:
        raise ValueError(f"Dataset {dataset_name} not found. Available datasets: {dataset_names}")
    DATASET_REGISTRY.fetch(f"{dataset_name}.zip", processor=pooch.Unzip(extract_dir=LOCAL_DATA_DIR), progressbar=progressbar)
    data_path = LOCAL_DATA_DIR / dataset_name
    return data_path.as_posix() + "/"
