"""Type aliases for anomaly detection.

This module provides type aliases used throughout the anomalib library. The aliases
include:
    - ``NORMALIZATION``: Type for normalization methods and configurations
    - ``THRESHOLD``: Type for threshold values and configurations

Example:
    >>> from anomalib.utils.types import NORMALIZATION, THRESHOLD
    >>> from anomalib.utils.normalization import NormalizationMethod
    >>> # Use min-max normalization
    >>> norm: NORMALIZATION = NormalizationMethod.MIN_MAX
    >>> print(norm)
    min_max
    >>> # Use threshold configuration
    >>> thresh: THRESHOLD = {"method": "adaptive", "delta": 0.1}

The module ensures consistent typing across the codebase and provides helpful type
hints for configuration objects.
"""

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
