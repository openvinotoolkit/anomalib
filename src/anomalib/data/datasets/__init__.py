"""PyTorch Dataset implementations for anomaly detection.

This module provides dataset implementations for various anomaly detection tasks:

Base Classes:
    - ``AnomalibDataset``: Base class for all Anomalib datasets
    - ``AnomalibDepthDataset``: Base class for 3D/depth datasets
    - ``AnomalibVideoDataset``: Base class for video datasets

Depth Datasets:
    - ``Folder3DDataset``: Custom RGB-D dataset from folder structure
    - ``MVTec3DDataset``: MVTec 3D AD dataset with industrial objects

Image Datasets:
    - ``BTechDataset``: BTech dataset containing industrial objects
    - ``DatumaroDataset``: Dataset in Datumaro format (Intel Geti™ export)
    - ``FolderDataset``: Custom dataset from folder structure
    - ``KolektorDataset``: Kolektor surface defect dataset
    - ``MVTecADDataset``: MVTec AD dataset with industrial objects
    - ``VAD``: Valeo Anomaly Detection Dataset
    - ``VisaDataset``: Visual Anomaly dataset

Video Datasets:
    - ``AvenueDataset``: CUHK Avenue dataset for abnormal event detection
    - ``ShanghaiTechDataset``: ShanghaiTech Campus surveillance dataset
    - ``UCSDpedDataset``: UCSD Pedestrian dataset for anomaly detection

Example:
    >>> from anomalib.data.datasets import MVTecADDataset
    >>> dataset = MVTecADDataset(
    ...     root="./datasets/MVTec",
    ...     category="bottle",
    ...     split="train"
    ... )
"""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .base import AnomalibDataset, AnomalibDepthDataset, AnomalibVideoDataset
from .depth import Folder3DDataset, MVTec3DDataset
from .image import (
    BTechDataset,
    DatumaroDataset,
    FolderDataset,
    KolektorDataset,
    MVTecADDataset,
    VADDataset,
    VisaDataset,
)
from .video import AvenueDataset, ShanghaiTechDataset, UCSDpedDataset

__all__ = [
    # Base
    "AnomalibDataset",
    "AnomalibDepthDataset",
    "AnomalibVideoDataset",
    # Depth
    "Folder3DDataset",
    "MVTec3DDataset",
    # Image
    "BTechDataset",
    "DatumaroDataset",
    "FolderDataset",
    "KolektorDataset",
    "MVTecADDataset",
    "VADDataset",
    "VisaDataset",
    # Video
    "AvenueDataset",
    "ShanghaiTechDataset",
    "UCSDpedDataset",
]
