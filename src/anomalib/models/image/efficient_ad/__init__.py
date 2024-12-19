"""EfficientAd: Accurate Visual Anomaly Detection at Millisecond-Level Latencies.

EfficientAd is a fast and accurate anomaly detection model that achieves
state-of-the-art performance with millisecond-level inference times. The model
utilizes a pre-trained EfficientNet backbone and employs a student-teacher
architecture for anomaly detection.

The implementation is based on the paper:
    "EfficientAd: Accurate Visual Anomaly Detection at Millisecond-Level Latencies"
    https://arxiv.org/pdf/2303.14535.pdf

Example:
    >>> from anomalib.models import EfficientAd
    >>> model = EfficientAd()
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .lightning_model import EfficientAd

__all__ = ["EfficientAd"]
