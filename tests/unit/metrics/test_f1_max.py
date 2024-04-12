"""Test F1Max metric."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from anomalib.metrics.f1_max import F1Max


def test_f1_max_logits() -> None:
    """Checks if F1Max metric computes the optimal F1 score.

    Test when the preds are in [0, 1]
    """
    metric = F1Max()

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


def test_f1_max_raw() -> None:
    """Checks if F1Max metric computes the optimal F1 score.

    Test when the preds are outside [0, 1]. BinaryPrecisionRecall automatically applies sigmoid.
    """
    metric = F1Max()

    preds = torch.tensor([-0.5, 0, 0.5, 1.0, 2])
    labels = torch.tensor([0, 1, 1, 1, 1])

    metric.update(preds, labels)
    assert metric.compute() == 1.0
    assert metric.threshold == 0.0

    metric.reset()
    preds = torch.tensor([-0.5, 0.0, 1.0, 2.0, -0.1])
    metric.update(preds, labels)
    assert metric.compute() == torch.tensor(1.0)
    assert metric.threshold == -0.1
