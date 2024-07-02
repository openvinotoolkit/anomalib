"""Tools for min-max normalization."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch


def normalize(
    targets: np.ndarray | np.float32 | torch.Tensor,
    threshold: float | np.ndarray | torch.Tensor,
    min_val: float | np.ndarray | torch.Tensor,
    max_val: float | np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Apply min-max normalization and shift the values such that the threshold value is centered at 0.5."""
    normalized = ((targets - threshold) / (max_val - min_val)) + 0.5
    if isinstance(targets, np.ndarray | np.float32 | np.float64):
        normalized = np.minimum(normalized, 1)
        normalized = np.maximum(normalized, 0)
    elif isinstance(targets, torch.Tensor):
        normalized = torch.minimum(normalized, torch.tensor(1))  # pylint: disable=not-callable
        normalized = torch.maximum(normalized, torch.tensor(0))  # pylint: disable=not-callable
    else:
        msg = f"Targets must be either Tensor or Numpy array. Received {type(targets)}"
        raise TypeError(msg)
    return normalized
