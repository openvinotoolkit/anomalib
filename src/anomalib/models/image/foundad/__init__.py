# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""FoundAD: Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors.

FoundAD trains a lightweight manifold projector on a frozen DINOv2 encoder with
CutPaste augmentation for off-manifold supervision. Anomaly detection is performed
by measuring the MSE between encoder features and projector output.

Example:
    >>> from anomalib.models.image import FoundAD
    >>> model = FoundAD()

Reference:
    Zhai et al., "Foundation Visual Encoders Are Secretly Few-Shot Anomaly
    Detectors", ICLR 2026. https://arxiv.org/abs/2510.01934
"""

from .lightning_model import FoundAD

__all__ = ["FoundAD"]
