# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""GRD-Net one-class anomaly detection with optional training ROI supervision.

The residual generator denoises synthetic anomalies, while an adversarial discriminator
and the existing DRÆM segmentator provide feature matching and localization. Predictions
average overlapping tiles and return a smoothed map and its maximum image score.

Reference:
    Ferrari, N., Fraccaroli, M., and Lamma, E. (2023).
    GRD-Net: Generative-Reconstructive-Discriminative Anomaly Detection with Region
    of Interest Attention Module. https://doi.org/10.1155/2023/7773481
"""

from .lightning_model import GRDNet

__all__ = ["GRDNet"]
