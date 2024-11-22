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

# A pooch download manager object
# Datasets are in the "data" subfolder as zip files, named as {dataset_name}_{suffix}.zip
# If the suffix is "_full", the zip file contains the full experimental dataset.
# For some of the larger datasets, a subset of the data is also available, which
# is stored in a zip file with the suffix "_subset".
dataset_manager = pooch.create(
    path=LOCAL_DATA_DIR,
    base_url=f"{DATA_URL}/data/",
    registry={
        "hafting_2008_subset.zip": "ca3b1bf417e496d81734b5d7beff93a9ac0c0c58660472db8c04c0ff097d6e2a",  # noqa: E501
        "hafting_2008_full.zip": "de778df50f21998c00f606208441a6d1e41ce25c4226e6abc013281690ff8dbe",  # noqa: E501
        "sargolini_2006_subset.zip": "0004511eeb416b7a78397d97cca540edd336a27a5326378a6d5809280fdc22b2",  # noqa: E501
        "sargolini_2006_full.zip": "99b03c1f71290c0a8381577f498fe2889ff111981e48792d8103fb1af375f44a",  # noqa: E501
        "wernle_2018_full.zip": "dfe6f15cdf617ce21267b5311756da2ab22b4fd021143d4dd795c85aad5da636",  # noqa: E501
    },
)


def find_datasets(download_manager: pooch.Pooch = dataset_manager) -> dict:
    """Find all available datasets in the remote data repository.

    Parameters
    ----------
    download_manager : pooch.Pooch
        A pooch download manager object that keeps track of available data.
        Default: neuralplayground.datasets.dataset_manager

    Returns
    -------
    dict
        A dictionary with dataset names as keys a list of available sizes as
        values.
    """
    sizes_per_dataset = {}

    for key in download_manager.registry.keys():
        file_name = key.split(".")[0]
        # Check that the file name ends with one of the expected suffixes
        size_suffixes = ["_full", "_subset"]
        assert any(
            [file_name.endswith(suffix) for suffix in size_suffixes]
        ), f"Dataset name {file_name} must end with one of the expected suffixes {size_suffixes}."
        # Extract the dataset name and the size suffix
        size_suffix = file_name.split("_")[-1]
        dataset_name = file_name.replace(f"_{size_suffix}", "")

        # Add the dataset name and size suffix to the dictionary
        if dataset_name not in sizes_per_dataset.keys():
            sizes_per_dataset[dataset_name] = []
        if size_suffix not in sizes_per_dataset[dataset_name]:
            sizes_per_dataset[dataset_name].append(size_suffix)

    return sizes_per_dataset


def fetch_data_path(
    dataset_name: str,
    subset: bool = True,
    progressbar: bool = True,
):
    """Download and cache a dataset from the GIN repository.

    Parameters
    ----------
    dataset_name : str
        The name of one the available datasets, e.g. "hafting_2008".
    subset : bool
        If True, see if a subset of the data is available and download that.
        If a subset is not available, or if set to False, download the full dataset.
        Defaults to True.
    progressbar : bool
        If True, show a progress bar while downloading the data.
        Defaults to True.

    Returns
    -------
    str
        Path to the downloaded dataset
    """
    sizes_per_dataset = find_datasets(dataset_manager)
    if dataset_name not in sizes_per_dataset:
        raise ValueError(f"Dataset {dataset_name} not found. Available datasets: {list(sizes_per_dataset.keys())}")

    file_name = f"{dataset_name}_full.zip"
    if subset:
        if "subset" in sizes_per_dataset[dataset_name]:
            file_name = f"{dataset_name}_subset.zip"
        else:
            print(f"Subset of dataset {dataset_name} not available. Downloading the full dataset instead.")

    dataset_manager.fetch(file_name, processor=pooch.Unzip(extract_dir=LOCAL_DATA_DIR), progressbar=progressbar)
    data_path = LOCAL_DATA_DIR / file_name.replace(".zip", "")
    return data_path.as_posix() + "/"
