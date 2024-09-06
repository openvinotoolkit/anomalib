"""Validate IO torch.Tensor data."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .image import validate_dimensions, validate_image
from .label import validate_batch_label, validate_label, validate_pred_label
from .mask import validate_mask, validate_pred_mask
from .score import (
    validate_anomaly_map,
    validate_pred_score,
)

__all__ = [
    # Common torch data item validations
    "validate_anomaly_map",
    "validate_dimensions",
    "validate_image",
    "validate_label",
    "validate_mask",
    "validate_pred_label",
    "validate_pred_mask",
    "validate_pred_score",
    # Torch batch validations
    "validate_batch_label",
]
