"""Torch Dataset Implementations of Anomalib Depth Datasets.

This module provides dataset implementations for working with RGB-D (depth) data in
anomaly detection tasks. The following datasets are available:

- ``Folder3DDataset``: Custom dataset for loading RGB-D data from a folder structure
- ``MVTec3DDataset``: Implementation of the MVTec 3D-AD dataset

Example:
    >>> from anomalib.data.datasets import Folder3DDataset
    >>> dataset = Folder3DDataset(
    ...     name="custom",
    ...     root="datasets/custom",
    ...     normal_dir="normal",
    ...     normal_depth_dir="normal_depth"
    ... )

    >>> from anomalib.data.datasets import MVTec3DDataset
    >>> dataset = MVTec3DDataset(
    ...     root="datasets/MVTec3D",
    ...     category="bagel"
    ... )
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .folder_3d import Folder3DDataset
from .mvtec_3d import MVTec3DDataset

__all__ = ["Folder3DDataset", "MVTec3DDataset"]
