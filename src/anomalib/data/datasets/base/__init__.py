"""Base Classes for Torch Datasets.

This module contains the base dataset classes used in anomalib for different data
modalities:

- ``AnomalibDataset``: Base class for image datasets
- ``AnomalibVideoDataset``: Base class for video datasets
- ``AnomalibDepthDataset``: Base class for depth/3D datasets

These classes extend PyTorch's Dataset class with additional functionality specific
to anomaly detection tasks.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .depth import AnomalibDepthDataset
from .image import AnomalibDataset
from .video import AnomalibVideoDataset

__all__ = ["AnomalibDataset", "AnomalibVideoDataset", "AnomalibDepthDataset"]
