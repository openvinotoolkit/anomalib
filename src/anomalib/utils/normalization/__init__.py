"""Tools for anomaly score normalization."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class NormalizationMethod(str, Enum):
    """Normalization method for normalization."""

    MIN_MAX = "min_max"
    NONE = "none"
