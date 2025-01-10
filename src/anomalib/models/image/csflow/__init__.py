"""Implementation of the CS-Flow model for anomaly detection.

The CS-Flow model, short for Cross-Scale-Flows, is a fully convolutional approach
for image-based defect detection. It leverages normalizing flows across multiple
scales of the input image to model the distribution of normal (non-defective)
samples.

The model architecture consists of:
    - A feature extraction backbone
    - Multiple normalizing flow blocks operating at different scales
    - Cross-scale connections to capture multi-scale dependencies

Example:
    >>> from anomalib.models.image.csflow import Csflow
    >>> model = Csflow()

Reference:
    Gudovskiy, Denis, et al. "Cflow-ad: Real-time unsupervised anomaly detection
    with localization via conditional normalizing flows."
    Proceedings of the IEEE/CVF Winter Conference on Applications of Computer
    Vision. 2022.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .lightning_model import Csflow

__all__ = ["Csflow"]
