"""Tests for the AUPRO metric."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from anomalib.metrics import AUPRO

from .aupro_reference import calculate_au_pro


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Generate fixtures for the tests."""
    labels = [
        torch.tensor(
            [
                [
                    [0, 0, 0, 1, 0, 0, 0],
                ]
                * 400,
            ],
        ),
        torch.tensor(
            [
                [
                    [0, 1, 0, 1, 0, 1, 0],
                ]
                * 400,
            ],
        ),
    ]
    preds = torch.arange(2800) / 2800.0
    preds = preds.view(1, 1, 400, 7)

    preds = [preds, preds]

    fpr_limit = [1 / 3, 1 / 3]
    expected_aupro = [torch.tensor(1 / 6), torch.tensor(1 / 6)]

    # Also test that per-region aupros are averaged
    labels.append(torch.cat(labels))
    preds.append(torch.cat(preds))
    fpr_limit.append(float(np.mean(fpr_limit)))
    expected_aupro.append(torch.tensor(np.mean(expected_aupro)))

    threshold_count = [
        200,
        200,
        200,
    ]

    if metafunc.function is test_aupro:
        vals = list(zip(labels, preds, fpr_limit, expected_aupro, strict=True))
        metafunc.parametrize(argnames=("labels", "preds", "fpr_limit", "expected_aupro"), argvalues=vals)
    elif metafunc.function is test_binned_aupro:
        vals = list(zip(labels, preds, threshold_count, strict=True))
        metafunc.parametrize(argnames=("labels", "preds", "threshold_count"), argvalues=vals)


def test_aupro(labels: torch.Tensor, preds: torch.Tensor, fpr_limit: float, expected_aupro: torch.Tensor) -> None:
    """Test if the AUPRO metric is computed correctly."""
    aupro = AUPRO(fpr_limit=fpr_limit)
    aupro.update(preds, labels)
    computed_aupro = aupro.compute()

    tmp_labels = [label.squeeze().numpy() for label in labels]
    tmp_preds = [pred.squeeze().numpy() for pred in preds]
    ref_aupro = torch.tensor(calculate_au_pro(tmp_labels, tmp_preds, integration_limit=fpr_limit)[0], dtype=torch.float)

    tolerance = 0.001
    assert torch.allclose(computed_aupro, expected_aupro, atol=tolerance)
    assert torch.allclose(computed_aupro, ref_aupro, atol=tolerance)


def test_binned_aupro(labels: torch.Tensor, preds: torch.Tensor, threshold_count: int) -> None:
    """Test if the binned aupro is the same as the non-binned aupro."""
    aupro = AUPRO()
    computed_not_binned_aupro = aupro(preds, labels)

    binned_pro = AUPRO(num_thresholds=threshold_count)
    computed_binned_aupro = binned_pro(preds, labels)

    tolerance = 0.001
    # with threshold binning the roc curve computed within the metric is more memory efficient
    # but a bit less accurate. So we check the difference in order to validate the binning effect.
    assert computed_binned_aupro != computed_not_binned_aupro
    assert torch.allclose(computed_not_binned_aupro, computed_binned_aupro, atol=tolerance)

    # test with prediction higher than 1
    preds = preds * 2
    computed_binned_aupro = binned_pro(preds, labels)
    computed_not_binned_aupro = aupro(preds, labels)

    # with threshold binning the roc curve computed within the metric is more memory efficient
    # but a bit less accurate. So we check the difference in order to validate the binning effect.
    assert computed_binned_aupro != computed_not_binned_aupro
    assert torch.allclose(computed_not_binned_aupro, computed_binned_aupro, atol=tolerance)
