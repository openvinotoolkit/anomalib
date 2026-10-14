# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test AUPR metric."""

import pytest
import torch
from torchmetrics.functional.classification import binary_average_precision

from anomalib.metrics.aupr import _AUPR as AUPR


def _aupr(preds: torch.Tensor, target: torch.Tensor, **kwargs: int) -> torch.Tensor:
    metric = AUPR(**kwargs)
    metric.update(preds, target)
    return metric.compute()


@pytest.fixture
def target() -> torch.Tensor:
    """2000 samples, 100 of them anomalous."""
    generator = torch.Generator().manual_seed(0)
    labels = torch.zeros(2000, dtype=torch.long)
    labels[torch.randperm(2000, generator=generator)[:100]] = 1
    return labels


@pytest.fixture
def random_preds(target: torch.Tensor) -> torch.Tensor:
    """Random scores with no relation to the labels."""
    return torch.rand(target.shape, generator=torch.Generator().manual_seed(1))


def test_aupr_perfect_ranking() -> None:
    """A detector that ranks every anomaly above every normal sample scores 1.0."""
    preds = torch.tensor([0.1, 0.2, 0.8, 0.9])
    target = torch.tensor([0, 0, 1, 1])
    assert _aupr(preds, target) == 1.0


def test_aupr_docstring_example() -> None:
    """The value in the ``_AUPR`` docstring is the average precision."""
    preds = torch.tensor([0.59, 0.35, 0.72, 0.33, 0.73, 0.81, 0.30, 0.05, 0.04, 0.48])
    target = torch.tensor([0, 1, 1, 1, 0, 0, 0, 0, 1, 1])
    assert _aupr(preds, target).round(decimals=4) == torch.tensor(0.4610)


@pytest.mark.parametrize("detector", ["constant", "all_zeros", "inverted"])
def test_aupr_uninformative_detectors_score_the_anomaly_rate(detector: str, target: torch.Tensor) -> None:
    """Detectors with no ranking information score the anomaly rate, not 0.525."""
    preds = {
        "constant": torch.full(target.shape, 0.5),
        "all_zeros": torch.zeros(target.shape),
        "inverted": 1.0 - target.float(),
    }[detector]
    assert torch.isclose(_aupr(preds, target), target.float().mean())


def test_aupr_matches_average_precision(target: torch.Tensor, random_preds: torch.Tensor) -> None:
    """AUPR agrees with the torchmetrics average precision."""
    assert torch.isclose(_aupr(random_preds, target), binary_average_precision(random_preds, target))


def test_aupr_binned_thresholds(target: torch.Tensor, random_preds: torch.Tensor) -> None:
    """Binned thresholds give a finite value that agrees with average precision."""
    result = _aupr(random_preds, target, thresholds=100)
    assert not torch.isnan(result)
    assert torch.isclose(result, binary_average_precision(random_preds, target, thresholds=100))


def test_aupr_incremental_update(target: torch.Tensor, random_preds: torch.Tensor) -> None:
    """Updating in batches gives the same value as one update."""
    metric = AUPR()
    for preds_batch, target_batch in zip(random_preds.chunk(4), target.chunk(4), strict=True):
        metric.update(preds_batch, target_batch)
    assert torch.isclose(metric.compute(), binary_average_precision(random_preds, target))
