"""Test binning."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from anomalib.metrics.binning import thresholds_between_0_and_1, thresholds_between_min_and_max


def test_thresholds_between_min_and_max() -> None:
    """Test if thresholds are between min and max."""
    preds = torch.Tensor([1, 10])
    assert torch.all(thresholds_between_min_and_max(preds, 2) == preds)


def test_thresholds_between_0_and_1() -> None:
    """Test if thresholds are between 0 and 1."""
    expected = torch.Tensor([0, 1])
    assert torch.all(thresholds_between_0_and_1(2) == expected)
