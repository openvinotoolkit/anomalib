"""Test OptimalF1 metric."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from anomalib.metrics.optimal_f1 import OptimalF1


def test_optimal_f1_logits() -> None:
    """Checks if OptimalF1 metric computes the optimal F1 score.

    Test when the preds are in [0, 1]
    """
    metric = OptimalF1()

    preds = torch.tensor([0.1, 0.5, 0.9, 1.0])
    labels = torch.tensor([0, 1, 1, 1])

    metric.update(preds, labels)
    assert metric.compute() == 1.0
    assert metric.threshold == 0.5

    metric.reset()
    preds = torch.tensor([0.1, 0.5, 0.9, 0.1])
    metric.update(preds, labels)
    # f1_score = (3 / 2) / (7 / 4 + 1e-10)  # noqa: ERA001
    assert metric.compute().round(decimals=4) == torch.tensor(0.8571)
    assert metric.threshold == 0.1


def test_optimal_f1_raw() -> None:
    """Checks if OptimalF1 metric computes the optimal F1 score.

    Test when the preds are outside [0, 1]. BinaryPrecisionRecall automatically applies sigmoid.
    """
    metric = OptimalF1()

    preds = torch.tensor([-0.5, 0, 0.5, 1.0, 2])
    labels = torch.tensor([0, 1, 1, 1, 1])

    metric.update(preds, labels)
    assert metric.compute() == 1.0
    assert metric.threshold == 0.5
