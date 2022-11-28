"""Methods to help post-process raw model outputs."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .normalization import NormalizationMethod
from .post_process import (
    ThresholdMethod,
    add_anomalous_label,
    add_normal_label,
    anomaly_map_to_color_map,
    compute_mask,
    superimpose_anomaly_map,
)
from .visualizer import ImageResult, Visualizer

__all__ = [
    "add_anomalous_label",
    "add_normal_label",
    "anomaly_map_to_color_map",
    "superimpose_anomaly_map",
    "compute_mask",
    "ImageResult",
    "NormalizationMethod",
    "Visualizer",
    "ThresholdMethod",
]
