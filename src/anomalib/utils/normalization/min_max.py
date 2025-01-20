"""Tools for min-max normalization.

This module provides utilities for min-max normalization of anomaly scores. The
main function :func:`normalize` scales values to [0,1] range and centers them
around a threshold.

Example:
    >>> import numpy as np
    >>> from anomalib.utils.normalization.min_max import normalize
    >>> # Create sample anomaly scores
    >>> scores = np.array([0.1, 0.5, 0.8])
    >>> threshold = 0.5
    >>> min_val = 0.0
    >>> max_val = 1.0
    >>> # Normalize scores
    >>> normalized = normalize(scores, threshold, min_val, max_val)
    >>> print(normalized)  # Values centered around 0.5
    [0.1 0.5 0.8]

The module supports both NumPy arrays and PyTorch tensors as inputs, with
appropriate handling for each type.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch


def normalize(
    targets: np.ndarray | np.float32 | torch.Tensor,
    threshold: float | np.ndarray | torch.Tensor,
    min_val: float | np.ndarray | torch.Tensor,
    max_val: float | np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Apply min-max normalization and center values around a threshold.

    This function performs min-max normalization on the input values and shifts them
    such that the threshold value is centered at 0.5. The output is clipped to the
    range [0,1].

    Args:
        targets (numpy.ndarray | numpy.float32 | torch.Tensor): Input values to
            normalize. Can be either a NumPy array or PyTorch tensor.
        threshold (float | numpy.ndarray | torch.Tensor): Threshold value that will
            be centered at 0.5 after normalization.
        min_val (float | numpy.ndarray | torch.Tensor): Minimum value used for
            normalization scaling.
        max_val (float | numpy.ndarray | torch.Tensor): Maximum value used for
            normalization scaling.

    Returns:
        numpy.ndarray | torch.Tensor: Normalized values in range [0,1] with
            threshold centered at 0.5. Output type matches input type.

    Raises:
        TypeError: If ``targets`` is neither a NumPy array nor PyTorch tensor.

    Example:
        >>> import torch
        >>> scores = torch.tensor([0.1, 0.5, 0.8])
        >>> threshold = 0.5
        >>> min_val = 0.0
        >>> max_val = 1.0
        >>> normalized = normalize(scores, threshold, min_val, max_val)
        >>> print(normalized)
        tensor([0.1000, 0.5000, 0.8000])
    """
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
