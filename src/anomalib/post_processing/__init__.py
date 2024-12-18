"""Post-processing module for anomaly detection results.

This module provides post-processing functionality for anomaly detection outputs:

- Base :class:`PostProcessor` class defining the post-processing interface
- :class:`OneClassPostProcessor` for one-class anomaly detection results

The post-processors handle:
    - Normalizing anomaly scores
    - Thresholding and anomaly classification
    - Mask generation and refinement
    - Result aggregation and formatting

Example:
    >>> from anomalib.post_processing import OneClassPostProcessor
    >>> post_processor = OneClassPostProcessor(threshold=0.5)
    >>> predictions = post_processor(anomaly_maps=anomaly_maps)
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .base import PostProcessor
from .one_class import OneClassPostProcessor

__all__ = ["OneClassPostProcessor", "PostProcessor"]
