"""Thresholding metrics for anomaly detection.

This module provides various thresholding techniques to convert anomaly scores into
binary predictions.

Available Thresholds:
    - ``BaseThreshold``: Abstract base class for implementing threshold methods
    - ``Threshold``: Generic threshold class that can be initialized with a value
    - ``F1AdaptiveThreshold``: Automatically finds optimal threshold by maximizing
      F1 score
    - ``ManualThreshold``: Allows manual setting of threshold value

Example:
    >>> from anomalib.metrics.threshold import ManualThreshold
    >>> threshold = ManualThreshold(threshold=0.5)
    >>> predictions = threshold(anomaly_scores=[0.1, 0.6, 0.3, 0.9])
    >>> print(predictions)
    [0, 1, 0, 1]
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .base import BaseThreshold, Threshold
from .f1_adaptive_threshold import F1AdaptiveThreshold
from .manual_threshold import ManualThreshold

__all__ = ["BaseThreshold", "Threshold", "F1AdaptiveThreshold", "ManualThreshold"]
