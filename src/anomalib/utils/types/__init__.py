"""Typing aliases for Anomalib."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import TypeAlias

from lightning.pytorch import Callback
from omegaconf import DictConfig, ListConfig

from anomalib.metrics.threshold import BaseThreshold
from anomalib.utils.normalization import NormalizationMethod

NORMALIZATION: TypeAlias = NormalizationMethod | DictConfig | Callback | str
THRESHOLD: TypeAlias = (
    BaseThreshold | tuple[BaseThreshold, BaseThreshold] | DictConfig | ListConfig | list[dict[str, str | float]] | str
)
