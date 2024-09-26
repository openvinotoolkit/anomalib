"""Per-Image Metrics."""

# Original Code
# https://github.com/jpcbertoldo/aupimo
#
# Modified
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .enums import StatsOutliersPolicy, StatsRepeatedPolicy, ThresholdMethod
from .pimo import AUPIMO, PIMO, AUPIMOResult, PIMOResult
from .utils import per_image_scores_stats

__all__ = [
    # constants
    "ThresholdMethod",
    "StatsOutliersPolicy",
    "StatsRepeatedPolicy",
    # result classes
    "PIMOResult",
    "AUPIMOResult",
    # torchmetrics interfaces
    "PIMO",
    "AUPIMO",
    # utils
    "per_image_scores_stats",
]
