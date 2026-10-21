# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Model-local data components for GRD-Net ROI supervision."""

from .dataclasses import GRDNetBatch, GRDNetItem
from .datamodule import GRDNetFolder
from .dataset import GRDNetFolderDataset

__all__ = ["GRDNetBatch", "GRDNetItem", "GRDNetFolder", "GRDNetFolderDataset"]
