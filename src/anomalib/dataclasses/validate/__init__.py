"""Validate IO data."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

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


def validate_anomaly_map(anomaly_map: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Validate the anomaly map."""
    return (
        validate_numpy_anomaly_map(anomaly_map)
        if isinstance(anomaly_map, np.ndarray)
        else validate_torch_anomaly_map(anomaly_map)
    )


def validate_dimensions(data: np.ndarray | torch.Tensor, expected_dims: int) -> np.ndarray | torch.Tensor:
    """Validate the dimensions of the data."""
    return (
        validate_numpy_dimensions(data, expected_dims)
        if isinstance(data, np.ndarray)
        else validate_torch_dimensions(data, expected_dims)
    )


def validate_image(image: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Validate the image."""
    if not isinstance(image, torch.Tensor):
        msg = "Numpy validation not implemented"
        raise NotImplementedError(msg)
    return validate_torch_image(image)


def validate_label(label: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Validate the label."""
    return validate_numpy_label(label) if isinstance(label, np.ndarray) else validate_torch_label(label)


def validate_mask(mask: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Validate the mask."""
    return validate_numpy_mask(mask) if isinstance(mask, np.ndarray) else validate_torch_mask(mask)


def validate_pred_label(pred_label: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Validate the predicted label."""
    return (
        validate_numpy_pred_label(pred_label)
        if isinstance(pred_label, np.ndarray)
        else validate_torch_pred_label(pred_label)
    )


def validate_pred_mask(pred_mask: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Validate the predicted mask."""
    return (
        validate_numpy_pred_mask(pred_mask)
        if isinstance(pred_mask, np.ndarray)
        else validate_torch_pred_mask(pred_mask)
    )


def validate_pred_score(pred_score: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Validate the predicted score."""
    return (
        validate_numpy_pred_score(pred_score)
        if isinstance(pred_score, np.ndarray)
        else validate_torch_pred_score(pred_score)
    )
