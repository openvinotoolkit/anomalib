"""Per-Image Metrics."""

# Original Code
# https://github.com/jpcbertoldo/aupimo
#
# Modified
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .binary_classification_curve import ThresholdMethod
from .pimo import AUPIMO, PIMO, AUPIMOResult, PIMOResult

__all__ = [
    # constants
    "ThresholdMethod",
    # result classes
    "PIMOResult",
    "AUPIMOResult",
    # torchmetrics interfaces
    "PIMO",
    "AUPIMO",
    "StatsOutliersPolicy",
]
