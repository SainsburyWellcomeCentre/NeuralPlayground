"""Module for fetching and loading saved and pre-trained models.

This module provides functions for fetching and loading pre-trained models used in tests,
examples, and tutorials. The pre-trained models are stored in a remote repository on GIN
and are downloaded to the user's local machine the first time they are used.
"""

import pooch

from neuralplayground.datasets import DATA_URL, LOCAL_DATA_DIR

# Create the local cache directory, if it doesn't exist
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

# A pooch download manager object
# Models are in the "data" subfolder as zip files - format: {model_name}.zip
model_manager = pooch.create(
    path=LOCAL_DATA_DIR,
    base_url=f"{DATA_URL}/data/",
    registry={
        "weber_2018_in_simple2D.zip": "0343782ccb06790fb639cd2e198971988d566eeac0aa41dcc4031665cbf53060",  # noqa: E501
        "weber_2018_in_hafting2008.zip": "af0e61373c69cbfcdb7a0ac0ce104b4039f12090197835055e1b1a8a88b53cfc",  # noqa: E501
        "weber_2018_in_sargolini2006.zip": "6a2577483eb116b7d910a05b58e5d26673ff7ff446469f19bde03e9a7675bf49",  # noqa: E501
        "stachenfeld_2018_in_simple2D.zip": "b4f5bd34728b9778e6643e431e90594083cf5ab7853b366d60e89a6f96022a5e",  # noqa: E501
        "stachenfeld_2018_in_sargolini2006.zip": "d619a66bcfb0fd4bdc7227912ef63dc35d27b92578ad766632893335318773b6",  # noqa: E501
        "stachenfeld_2018_in_hafting2008.zip": "960cdc8d4fa9ef86ed1d5ef144fe6949d227c081b837ae24e49335bdaf971899",  # noqa: E501
        "weber_2018_in_wernle.zip": "51f701966229ba8a70aab7b7ce79f4965e80904661eb6cdad85d03b0ddb7f8ff",  # noqa: E501
        "weber_2018_in_merging_room.zip": "10c537bc1d410de1bba18fe36624501bc4caddc0a032f3889a39435256a0205c",  # noqa: E501
        "whittington_2020_in_discritized_objects.zip": "3b527b03cd011b5e71ff66304f25d2406acddcbd3f770139ca8d8edc71cf1703",  # noqa: E501
    },
)


def find_saved_models(download_manager: pooch.Pooch = model_manager) -> list:
    """Find all available saved models in the remote data repository.

    Parameters
    ----------
    download_manager : pooch.Pooch
        A pooch download manager object that keeps track of available data.
        Default: neuralplayground.saved_models.model_manager

    Returns
    -------
    list
        A list of available model names.
    """
    available_models = [model_name.split(".")[0] for model_name in download_manager.registry.keys()]
    return available_models


def fetch_model_path(
    model_name: str,
    progressbar: bool = True,
):
    """Download and cache a model from the GIN repository.

    Parameters
    ----------
    model_name : str
        The name of one the available model, e.g. "hafting_2008".
    progressbar : bool
        If True, show a progress bar while downloading the model.
        Defaults to True.

    Returns
    -------
    str
        Path to the downloaded model
    """
    available_models = find_saved_models(model_manager)
    if model_name not in available_models:
        raise ValueError(f"Model {model_name} not found. Available models: {available_models}")
    model_manager.fetch(f"{model_name}.zip", processor=pooch.Unzip(extract_dir=LOCAL_DATA_DIR), progressbar=progressbar)
    model_path = LOCAL_DATA_DIR / model_name
    return model_path.as_posix() + "/"
