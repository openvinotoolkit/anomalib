"""Tests custom precision recall metric."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from anomalib.metrics import BinaryPrecisionRecallCurve


def test_precision_recall() -> None:
    """Test if the precision recall computation returns the desired value."""
    targets = torch.tensor([0, 0, 1, 0, 1, 1], dtype=torch.int)
    predictions = torch.tensor([12, 13, 14, 15, 16, 17])
    metric = BinaryPrecisionRecallCurve()
    metric.update(predictions, targets)
    _, _, thresholds = metric.compute()
    assert (thresholds == torch.tensor([12, 13, 14, 15, 16, 17])).all()
