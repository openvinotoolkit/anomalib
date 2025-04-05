"""Tests for the adaptive threshold metric."""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from anomalib.metrics.threshold.f1_adaptive_threshold import _F1AdaptiveThreshold


@pytest.mark.parametrize(
    ("labels", "preds", "target_threshold"),
    [
        (torch.Tensor([0, 0, 0, 1, 1]), torch.Tensor([2.3, 1.6, 2.6, 7.9, 3.3]), 3.3),  # standard case
        (torch.Tensor([1, 0, 0, 0]), torch.Tensor([4, 3, 2, 1]), 4),  # 100% recall for all thresholds
        (torch.Tensor([1, 1, 1, 1]), torch.Tensor([4, 3, 2, 1]), 1),  # use minimum value when all images are anomalous
        (torch.Tensor([0, 0, 0, 0]), torch.Tensor([4, 3, 2, 1]), 4),  # use maximum value when all images are normal
    ],
)
def test_adaptive_threshold(labels: torch.Tensor, preds: torch.Tensor, target_threshold: int | float) -> None:
    """Test if the adaptive threshold computation returns the desired value."""
    adaptive_threshold = _F1AdaptiveThreshold()
    adaptive_threshold.update(preds, labels)
    threshold_value = adaptive_threshold.compute()

    assert threshold_value == target_threshold
