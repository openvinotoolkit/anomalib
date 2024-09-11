"""Base Classes for Torch Datasets."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .depth import AnomalibDepthDataset
from .image import AnomalibDataset
from .video import AnomalibVideoDataset

__all__ = ["AnomalibDataset", "AnomalibVideoDataset", "AnomalibDepthDataset"]
