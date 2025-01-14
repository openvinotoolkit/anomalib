"""PyTorch Dataset implementations for anomaly detection in images.

This module provides dataset implementations for various image anomaly detection
datasets:

- ``BTechDataset``: BTech dataset containing industrial objects
- ``DatumaroDataset``: Dataset in Datumaro format (Intel Geti™ export)
- ``FolderDataset``: Custom dataset from folder structure
- ``KolektorDataset``: Kolektor surface defect dataset
- ``MVTecDataset``: MVTec AD dataset with industrial objects
- ``VisaDataset``: Visual Inspection of Surface Anomalies dataset

Example:
    >>> from anomalib.data.datasets import MVTecDataset
    >>> dataset = MVTecDataset(
    ...     root="./datasets/MVTec",
    ...     category="bottle",
    ...     split="train"
    ... )
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .btech import BTechDataset
from .datumaro import DatumaroDataset
from .folder import FolderDataset
from .kolektor import KolektorDataset
from .mvtec import MVTecDataset
from .visa import VisaDataset

__all__ = [
    "BTechDataset",
    "DatumaroDataset",
    "FolderDataset",
    "KolektorDataset",
    "MVTecDataset",
    "VisaDataset",
]
