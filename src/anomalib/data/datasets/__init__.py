# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch Dataset implementations for anomaly detection.

This module provides dataset implementations for various anomaly detection tasks:

Base Classes:
    - ``AnomalibDataset``: Base class for all Anomalib datasets
    - ``AnomalibDepthDataset``: Base class for 3D/depth datasets
    - ``AnomalibVideoDataset``: Base class for video datasets

Depth Datasets:
    - ``Folder3DDataset``: Custom RGB-D dataset from folder structure
    - ``MVTec3DDataset``: MVTec 3D AD dataset with industrial objects
    - ``ADAM3DDataset``: 3D ADAM dataset with additive manufactured objects

Image Datasets:
    - ``AutoVIDataset``: AutoVI Dataset for Automotive Visual Defect Inspection
    - ``BTechDataset``: BTech dataset containing industrial objects
    - ``DatumaroDataset``: Dataset in Datumaro format (Intel Geti™ export)
    - ``FolderDataset``: Custom dataset from folder structure
    - ``KolektorDataset``: Kolektor surface defect dataset
    - ``MPDDDataset``: Metal Parts Defect Detection dataset
    - ``MVTecADDataset``: MVTec AD dataset with industrial objects
    - ``TabularDataset``: Custom tabular dataset with image paths and labels
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

from .base import AnomalibDataset, AnomalibDepthDataset, AnomalibVideoDataset
from .depth import ADAM3DDataset, Folder3DDataset, MVTec3DDataset
from .image import (
    AutoVIDataset,
    BMADDataset,
    BTechDataset,
    DatumaroDataset,
    FolderDataset,
    KolektorDataset,
    MPDDDataset,
    MVTecADDataset,
    TabularDataset,
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
    "ADAM3DDataset",
    # Image
    "AutoVIDataset",
    "BMADDataset",
    "BTechDataset",
    "DatumaroDataset",
    "FolderDataset",
    "KolektorDataset",
    "MPDDDataset",
    "MVTecADDataset",
    "TabularDataset",
    "VADDataset",
    "VisaDataset",
    # Video
    "AvenueDataset",
    "ShanghaiTechDataset",
    "UCSDpedDataset",
]
