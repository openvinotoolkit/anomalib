"""Typing aliases for Anomalib."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import TypeAlias

from lightning.pytorch import Callback
from omegaconf import DictConfig, ListConfig

from anomalib.metrics.threshold import Threshold
from anomalib.utils.normalization import NormalizationMethod

NORMALIZATION: TypeAlias = NormalizationMethod | DictConfig | Callback | str
THRESHOLD: TypeAlias = (
    Threshold | tuple[Threshold, Threshold] | DictConfig | ListConfig | list[dict[str, str | float]] | str
)
