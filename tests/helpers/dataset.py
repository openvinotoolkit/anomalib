import os
from pathlib import Path
from typing import Union


def get_dataset_path(path: Union[str, Path] = "./datasets/MVTec"):
    """
    Selects path based on tests in local system or docker image.
    Local install assumes dataset is downloaded to anomaly/datasets/MVTec.
    In either case, if the location is empty, the dataset is downloaded again.
    This speeds up tests in docker images where dataset is already stored in /tmp/anomalib
    """
    # when running locally
    path = str(path)
    if not os.path.isdir(path):
        # when using docker image
        path = "/tmp/anomalib/datasets/MVTec"
    return path
