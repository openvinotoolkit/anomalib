"""Test MinMax metric."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from anomalib.metrics.min_max import _MinMax  # Assuming the metric is part of `anomalib`


def test_initialization() -> None:
    """Test if the metric initializes with correct default values."""
    metric = _MinMax()
    assert torch.isinf(metric.min), "Initial min should be positive infinity."
    assert metric.min > 0, "Initial min should be positive infinity."
    assert torch.isinf(metric.max), "Initial max should be negative infinity."
    assert metric.max < 0, "Initial max should be negative infinity."


def test_update_single_batch() -> None:
    """Test updating the metric with a single batch."""
    metric = _MinMax()
    batch = torch.tensor([1.0, 2.0, 3.0, -1.0])
    metric.update(batch)

    assert metric.min.item() == -1.0, "Min should be -1.0 after single batch update."
    assert metric.max.item() == 3.0, "Max should be 3.0 after single batch update."


def test_update_multiple_batches() -> None:
    """Test updating the metric with multiple batches."""
    metric = _MinMax()
    batch1 = torch.tensor([0.5, 1.5, 3.0])
    batch2 = torch.tensor([-0.5, 0.0, 2.5])

    metric.update(batch1)
    metric.update(batch2)

    assert metric.min.item() == -0.5, "Min should be -0.5 after multiple batch updates."
    assert metric.max.item() == 3.0, "Max should be 3.0 after multiple batch updates."


def test_compute() -> None:
    """Test computation of the min and max values after updates."""
    metric = _MinMax()
    batch1 = torch.tensor([1.0, 2.0])
    batch2 = torch.tensor([-1.0, 0.0])

    metric.update(batch1)
    metric.update(batch2)

    min_val, max_val = metric.compute()

    assert min_val.item() == -1.0, "Computed min should be -1.0."
    assert max_val.item() == 2.0, "Computed max should be 2.0."


def test_no_updates() -> None:
    """Test behavior when no updates are made to the metric."""
    metric = _MinMax()

    min_val, max_val = metric.compute()

    assert torch.isinf(min_val), "Min should remain positive infinity with no updates."
    assert min_val > 0, "Min should remain positive infinity with no updates."
    assert torch.isinf(max_val), "Max should remain negative infinity with no updates."
    assert max_val < 0, "Max should remain negative infinity with no updates."
