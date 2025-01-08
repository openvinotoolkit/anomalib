"""Per-Image Metrics for anomaly detection.

This module provides metrics for evaluating anomaly detection performance on a
per-image basis. The metrics include:

- ``PIMO``: Per-Image Metric Optimization for anomaly detection
- ``AUPIMO``: Area Under PIMO curve
- ``ThresholdMethod``: Methods for determining optimal thresholds
- ``PIMOResult``: Container for PIMO metric results
- ``AUPIMOResult``: Container for AUPIMO metric results

The implementation is based on the original work from:
https://github.com/jpcbertoldo/aupimo

Example:
    >>> from anomalib.metrics.pimo import PIMO, AUPIMO
    >>> pimo = PIMO()  # doctest: +SKIP
    >>> aupimo = AUPIMO()  # doctest: +SKIP
"""

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
