"""Binning functions for metrics.

This module provides utility functions for generating threshold values used in
various metrics calculations.

Example:
    >>> import torch
    >>> from anomalib.metrics.binning import thresholds_between_min_and_max
    >>> preds = torch.tensor([0.1, 0.5, 0.8])
    >>> thresholds = thresholds_between_min_and_max(preds, num_thresholds=3)
    >>> thresholds
    tensor([0.1000, 0.4500, 0.8000])

    Generate thresholds between 0 and 1:
    >>> from anomalib.metrics.binning import thresholds_between_0_and_1
    >>> thresholds = thresholds_between_0_and_1(num_thresholds=3)
    >>> thresholds
    tensor([0.0000, 0.5000, 1.0000])
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import linspace


def thresholds_between_min_and_max(
    preds: torch.Tensor,
    num_thresholds: int = 100,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate evenly spaced threshold values between min and max predictions.

    Args:
        preds (torch.Tensor): Input tensor containing predictions or scores.
        num_thresholds (int, optional): Number of threshold values to generate.
            Defaults to ``100``.
        device (torch.device | None, optional): Device on which to place the
            output tensor. If ``None``, uses the device of input tensor.
            Defaults to ``None``.

    Returns:
        torch.Tensor: A 1D tensor of size ``num_thresholds`` containing evenly
            spaced values between ``preds.min()`` and ``preds.max()``.

    Example:
        >>> preds = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        >>> thresholds = thresholds_between_min_and_max(preds, num_thresholds=3)
        >>> thresholds
        tensor([0.1000, 0.5000, 0.9000])
    """
    return linspace(start=preds.min(), end=preds.max(), steps=num_thresholds, device=device)


def thresholds_between_0_and_1(
    num_thresholds: int = 100,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate evenly spaced threshold values between 0 and 1.

    Args:
        num_thresholds (int, optional): Number of threshold values to generate.
            Defaults to ``100``.
        device (torch.device | None, optional): Device on which to place the
            output tensor. Defaults to ``None``.

    Returns:
        torch.Tensor: A 1D tensor of size ``num_thresholds`` containing evenly
            spaced values between ``0`` and ``1``.

    Example:
        >>> thresholds = thresholds_between_0_and_1(num_thresholds=5)
        >>> thresholds
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
    """
    return linspace(start=0, end=1, steps=num_thresholds, device=device)
