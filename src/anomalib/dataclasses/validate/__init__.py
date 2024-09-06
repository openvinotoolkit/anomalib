"""Validate IO data."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Numpy validation imports
from .ndarray import validate_anomaly_map as validate_numpy_anomaly_map
from .ndarray import validate_dimensions as validate_numpy_dimensions
from .ndarray import validate_label as validate_numpy_label
from .ndarray import validate_mask as validate_numpy_mask
from .ndarray import validate_pred_label as validate_numpy_pred_label
from .ndarray import validate_pred_mask as validate_numpy_pred_mask
from .ndarray import validate_pred_score as validate_numpy_pred_score

# Path validation imports
from .path import validate_path

# Torch validation imports
from .tensor import validate_anomaly_map as validate_torch_anomaly_map
from .tensor import validate_dimensions as validate_torch_dimensions
from .tensor import validate_image as validate_torch_image
from .tensor import validate_label as validate_torch_label
from .tensor import validate_mask as validate_torch_mask
from .tensor import validate_pred_label as validate_torch_pred_label
from .tensor import validate_pred_mask as validate_torch_pred_mask
from .tensor import validate_pred_score as validate_torch_pred_score

__all__ = [
    # Path validation functions
    "validate_path",
    # Numpy validation functions
    "validate_numpy_anomaly_map",
    "validate_numpy_dimensions",
    "validate_numpy_label",
    "validate_numpy_mask",
    "validate_numpy_pred_label",
    "validate_numpy_pred_mask",
    "validate_numpy_pred_score",
    # Torch validation functions
    "validate_torch_anomaly_map",
    "validate_torch_dimensions",
    "validate_torch_image",
    "validate_torch_label",
    "validate_torch_mask",
    "validate_torch_pred_label",
    "validate_torch_pred_mask",
    "validate_torch_pred_score",
]
