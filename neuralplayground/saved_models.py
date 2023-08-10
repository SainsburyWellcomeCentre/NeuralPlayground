"""Module for fetching and loading saved and pre-trainned models.

This module provides functions for fetching and loading pre-trainned model used in test,
examples, and tutorials. The pre-trainned model are stored in a remote repository on GIN
and are downloaded to the user's local machine the first time they are used.
"""

from pathlib import Path

import pooch

# URL to GIN model repository where the experimental model are hosted
Model_URL = "https://gin.g-node.org/SainsburyWellcomeCentre/NeuralPlayground/raw/master"

# Data to be downloaded and cached in ~/.NeuralPlayground/data
LOCAL_MODEL_DIR = Path("~", ".NeuralPlayground", "data").expanduser()
LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# A pooch data registry object
# model are in the "data" subfolder as zip files - format: {model_name}.zip
MODEL_REGISTRY = pooch.create(
    path=LOCAL_MODEL_DIR,
    base_url=f"{Model_URL}/data/",
    registry={
        "weber_2018_in_simple2D.zip": "85028da0cc7d657933b0041458e9b61149ad86142d370808eabdd3478f8c1cca",
        "weber_2018_in_hafting2008.zip": "7bc345632c8cb45b04ed7be738afd3b740259d3d3fb1973d441e920ceeb66e1d",
        "weber_2018_in_sargolini2006.zip": "3c3ac58a7ef45a560a8a92b49c31e7724f6d5610c223de6fcf80ed1200a410ab",
        "stachenfeld_2018_in_simple2D.zip": "b0d2db8e2b5ee2d7a4f9b8ea1ba1366dafbbe72bb213363a8536fe930649487e",
        "stachenfeld_2018_in_sargolini2006.zip": "7cadfbe2024d5e3ccbae13f9a2ba492791c3a5f82f740e6289c5eadd3d8d63d3",
        "stachenfeld_2018_in_hafting2008.zip": "f014be0d6f399b50ebc67e1c4530db24f76ce12fa210aeff2695717598517137",
        "weber_2018_in_wernle.zip": "78770840f56e8c7fb22182aa2530027e44be5a9eb9127cdcd291d86bbb3e6623",
        "weber_2018_in_merging_room.zip": "2aed903a302d0965637c42ab36a9547d2e824124872595af6ba1d763cc381531",
        # noqa: E501# noqa: E501
    },
)


model_names = [n.split(".")[0] for n in MODEL_REGISTRY.registry.keys()]


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
    if model_name not in model_names:
        raise ValueError(f"Model {model_name} not found. Available models: {model_names}")
    MODEL_REGISTRY.fetch(f"{model_name}.zip", processor=pooch.Unzip(extract_dir=LOCAL_MODEL_DIR), progressbar=progressbar)
    model_path = LOCAL_MODEL_DIR / model_name
    return model_path.as_posix() + "/"
