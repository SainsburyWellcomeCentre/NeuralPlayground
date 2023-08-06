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
        "weber_2018_in_simple2D.zip": "85028da0cc7d657933b0041458e9b61149ad86142d370808eabdd3478f8c1cca",
        "weber_2018_in_Hafting2008.zip": "7bc345632c8cb45b04ed7be738afd3b740259d3d3fb1973d441e920ceeb66e1d",
        "weber_2018_in_Sargolini2006.zip": "3c3ac58a7ef45a560a8a92b49c31e7724f6d5610c223de6fcf80ed1200a410ab",
        "stachenfeld_2018_in_simple2D.zip": "b0d2db8e2b5ee2d7a4f9b8ea1ba1366dafbbe72bb213363a8536fe930649487e",
        "stachenfeld_2018_in_Sargolini2006.zip": "7cadfbe2024d5e3ccbae13f9a2ba492791c3a5f82f740e6289c5eadd3d8d63d3",
        "stachenfeld_2018_in_Hafting2008.zip": "afd6104b295477acfc2cbbcfe5c5cfd711d426516194e6faeec8a83117d45e3e",
        "weber_2018_in_Wernle.zip": "afd6104b295477acfc2cbbcfe5c5cfd711d426516194e6faeec8a83117d45e3e",
"weber_2018_in_Merging_Room.zip": "2aed903a302d0965637c42ab36a9547d2e824124872595af6ba1d763cc381531",
        # noqa: E501# noqa: E501
    },
)


dataset_names = [n.split(".")[0] for n in DATASET_REGISTRY.registry.keys()]


def fetch_model_path(
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
