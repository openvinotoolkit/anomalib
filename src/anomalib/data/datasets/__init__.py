"""Torch Dataset Implementations of Anomalib Datasets."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .base import AnomalibDataset, AnomalibDepthDataset, AnomalibVideoDataset
from .depth import Folder3DDataset, MVTec3DDataset
from .image import BTechDataset, CSVDataset, DatumaroDataset, FolderDataset, KolektorDataset, MVTecDataset, VisaDataset
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
    "CSVDataset",
    "DatumaroDataset",
    "FolderDataset",
    "KolektorDataset",
    "MVTecDataset",
    "VisaDataset",
    # Video
    "AvenueDataset",
    "ShanghaiTechDataset",
    "UCSDpedDataset",
]
