"""Tests for the AUPRO metric."""

import numpy as np
import torch

from anomalib.utils.metrics import AUPRO
from tests.helpers.aupro_reference import calculate_au_pro


def pytest_generate_tests(metafunc):
    labels = [
        torch.tensor(
            [
                [
                    [0, 0, 0, 1, 0, 0, 0],
                ]
                * 400,
            ]
        ),
        torch.tensor(
            [
                [
                    [0, 1, 0, 1, 0, 1, 0],
                ]
                * 400,
            ]
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
        vals = list(zip(labels, preds, fpr_limit, expected_aupro))
        metafunc.parametrize(argnames=("labels", "preds", "fpr_limit", "expected_aupro"), argvalues=vals)
    elif metafunc.function is test_binned_aupro:
        vals = list(zip(labels, preds, threshold_count))
        metafunc.parametrize(argnames=("labels", "preds", "threshold_count"), argvalues=vals)


def test_aupro(labels, preds, fpr_limit, expected_aupro):
    aupro = AUPRO(fpr_limit=fpr_limit)
    aupro.update(preds, labels)
    computed_aupro = aupro.compute()

    tmp_labels = [label.squeeze().numpy() for label in labels]
    tmp_preds = [pred.squeeze().numpy() for pred in preds]
    ref_aupro = torch.tensor(calculate_au_pro(tmp_labels, tmp_preds, integration_limit=fpr_limit)[0], dtype=torch.float)

    TOL = 0.001
    assert torch.allclose(computed_aupro, expected_aupro, atol=TOL)
    assert torch.allclose(computed_aupro, ref_aupro, atol=TOL)


def test_binned_aupro(labels, preds, threshold_count):
    aupro = AUPRO()
    computed_not_binned_aupro = aupro(preds, labels)

    binned_pro = AUPRO(num_thresholds=threshold_count)
    computed_binned_aupro = binned_pro(preds, labels)

    TOL = 0.001
    # with threshold binning the roc curve computed within the metric is more memory efficient
    # but a bit less accurate. So we check the difference in order to validate the binning effect.
    assert computed_binned_aupro != computed_not_binned_aupro
    assert torch.allclose(computed_not_binned_aupro, computed_binned_aupro, atol=TOL)

    # test with prediction higher than 1
    preds = preds * 2
    computed_binned_aupro = binned_pro(preds, labels)
    computed_not_binned_aupro = aupro(preds, labels)

    # with threshold binning the roc curve computed within the metric is more memory efficient
    # but a bit less accurate. So we check the difference in order to validate the binning effect.
    assert computed_binned_aupro != computed_not_binned_aupro
    assert torch.allclose(computed_not_binned_aupro, computed_binned_aupro, atol=TOL)
