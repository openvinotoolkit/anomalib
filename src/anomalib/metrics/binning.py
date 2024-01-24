"""Binning functions for metrics."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import linspace


def thresholds_between_min_and_max(
    preds: torch.Tensor,
    num_thresholds: int = 100,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Threshold values between min and max of the predictions.

    Args:
        preds (torch.Tensor): Predictions.
        num_thresholds (int, optional): Number of thresholds to generate. Defaults to 100.
        device (torch_device | None, optional): Device to use for computation. Defaults to None.

    Returns:
        Tensor:
            Array of size ``num_thresholds`` that contains evenly spaced values
            between ``preds.min()`` and ``preds.max()`` on ``device``.
    """
    return linspace(start=preds.min(), end=preds.max(), steps=num_thresholds, device=device)


def thresholds_between_0_and_1(num_thresholds: int = 100, device: torch.device | None = None) -> torch.Tensor:
    """Threshold values between 0 and 1.

    Args:
        num_thresholds (int, optional): Number of thresholds to generate. Defaults to 100.
        device (torch_device | None, optional): Device to use for computation. Defaults to None.

    Returns:
        Tensor: Threshold values between 0 and 1.
    """
    return linspace(start=0, end=1, steps=num_thresholds, device=device)
