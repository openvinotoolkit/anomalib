"""Implementation of the AI-VAD model.

This module provides the implementation of the AI-VAD
Attribute-based Representations for Accurate and Interpretable Video Anomaly
Detection.

The model extracts three types of features from video regions:
    - Velocity features: Histogram of optical flow magnitudes
    - Pose features: Human keypoint detections using KeypointRCNN
    - Deep features: CLIP embeddings of region crops

These features are used to model normal behavior patterns and detect anomalies as
deviations from the learned distributions.

Example:
    >>> from anomalib.models.video.ai_vad import AiVad
    >>> # Initialize the model
    >>> model = AiVad(
    ...     input_size=(256, 256),
    ...     use_pose_features=True,
    ...     use_deep_features=True,
    ...     use_velocity_features=True
    ... )

Reference:
    Tal Reiss, Yedid Hoshen, "AI-VAD: Attribute-based Representations for
    Accurate and Interpretable Video Anomaly Detection", arXiv:2212.00789, 2022
    https://arxiv.org/pdf/2212.00789.pdf
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .lightning_model import AiVad

__all__ = ["AiVad"]
