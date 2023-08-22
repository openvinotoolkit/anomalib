"""EfficientAd: Accurate Visual Anomaly Detection at Millisecond-Level Latencies.
https://arxiv.org/pdf/2303.14535.pdf
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .lightning_model import EfficientAd, EfficientAdLightning

__all__ = ["EfficientAd", "EfficientAdLightning"]
