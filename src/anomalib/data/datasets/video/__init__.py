"""PyTorch Dataset implementations for anomaly detection in videos.

This module provides dataset implementations for various video anomaly detection
datasets:

- ``AvenueDataset``: CUHK Avenue dataset for abnormal event detection
- ``ShanghaiTechDataset``: ShanghaiTech Campus surveillance dataset
- ``UCSDpedDataset``: UCSD Pedestrian dataset for anomaly detection

Example:
    >>> from anomalib.data.datasets import AvenueDataset
    >>> dataset = AvenueDataset(
    ...     root="./datasets/avenue",
    ...     split="train"
    ... )
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .avenue import AvenueDataset
from .shanghaitech import ShanghaiTechDataset
from .ucsd_ped import UCSDpedDataset

__all__ = ["AvenueDataset", "ShanghaiTechDataset", "UCSDpedDataset"]
