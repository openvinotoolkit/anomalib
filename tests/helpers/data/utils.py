"""Tests - Helpers Utils."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os


def get_dataset_path(dataset: str = "MVTec") -> str:
    """Selects path based on tests in local system or docker image.

    Local install assumes datasets are located in anomaly/datasets/.
    In either case, if the location is empty, the dataset is downloaded again.
    This speeds up tests in docker images where dataset is already stored in /tmp/anomalib

    Example:
    Assume that `datasets directory exists in ~/anomalib/,

    >>> get_dataset_path(dataset="MVTec")
    './datasets/MVTec'

    """
    # Initially check if `datasets` directory exists locally and look
    # for the `dataset`. This is useful for local testing.
    path = os.path.join("./datasets", dataset)

    # For docker deployment or a CI that runs on server, dataset directory
    # may not necessarily be located in the repo. Therefore, check anomalib
    # dataset path environment variable.
    if not os.path.isdir(path):
        path = os.path.join(os.environ["ANOMALIB_DATASET_PATH"], dataset)
    return path
