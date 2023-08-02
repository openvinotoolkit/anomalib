"""Tools for min-max normalization."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def normalize(
    targets: np.ndarray | np.float32 | Tensor,
    threshold: float | np.ndarray | Tensor,
    min_val: float | np.ndarray | Tensor,
    max_val: float | np.ndarray | Tensor,
) -> np.ndarray | Tensor:
    """Apply min-max normalization and shift the values such that the threshold value is centered at 0.5."""
    normalized = ((targets - threshold) / (max_val - min_val)) + 0.5
    if isinstance(targets, (np.ndarray, np.float32, np.float64)):
        normalized = np.minimum(normalized, 1)
        normalized = np.maximum(normalized, 0)
    elif isinstance(targets, Tensor):
        normalized = torch.minimum(normalized, torch.tensor(1))  # pylint: disable=not-callable
        normalized = torch.maximum(normalized, torch.tensor(0))  # pylint: disable=not-callable
    else:
        raise ValueError(f"Targets must be either Tensor or Numpy array. Received {type(targets)}")
    return normalized
