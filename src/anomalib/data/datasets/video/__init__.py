"""Torch Dataset Implementations of Anomalib Video Datasets."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .avenue import AvenueDataset
from .shanghaitech import ShanghaiTechDataset
from .ucsd_ped import UCSDpedDataset

__all__ = ["AvenueDataset", "ShanghaiTechDataset", "UCSDpedDataset"]
