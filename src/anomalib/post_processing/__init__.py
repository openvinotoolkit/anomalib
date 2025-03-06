"""Post-processing module for anomaly detection results.

This module provides post-processing functionality for anomaly detection outputs:

- :class:`PostProcessor` for one-class anomaly detection results

The post-processors handle:
    - Normalizing anomaly scores
    - Thresholding and anomaly classification
    - Mask generation and refinement
    - Result aggregation and formatting

Example:
    >>> from anomalib.post_processing import PostProcessor
    >>> post_processor = PostProcessor(threshold=0.5)
    >>> predictions = post_processor(anomaly_maps=anomaly_maps)
"""

# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .post_processor import PostProcessor

__all__ = ["PostProcessor"]
