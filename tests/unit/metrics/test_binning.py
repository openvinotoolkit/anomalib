# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test binning."""

import torch

from anomalib.metrics.binning import thresholds_between_0_and_1, thresholds_between_min_and_max


def test_thresholds_between_min_and_max() -> None:
    """Test if thresholds are between min and max."""
    preds = torch.tensor([1, 10])
    assert torch.all(thresholds_between_min_and_max(preds, 2) == preds)


def test_thresholds_between_0_and_1() -> None:
    """Test if thresholds are between 0 and 1."""
    expected = torch.tensor([0, 1])
    assert torch.all(thresholds_between_0_and_1(2) == expected)


def test_thresholds_between_min_and_max_device() -> None:
    """Test if thresholds are created on the correct device."""
    preds = torch.tensor([1, 10])  # cpu

    # Test fallback to preds.device when device=None
    with torch.device("meta"):
        out = thresholds_between_min_and_max(preds, 2)
    assert out.device.type == "cpu"

    # Test explicit device works
    out_explicit = thresholds_between_min_and_max(preds, 2, device=torch.device("meta"))
    assert out_explicit.device.type == "meta"
