"""Tests for the AUPRO metric."""

import numpy as np
import torch

from anomalib.utils.metrics import AUPRO
from tests.helpers.aupro_reference import calculate_au_pro


def pytest_generate_tests(metafunc):
    if metafunc.function is test_pro:
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
        aupro = [torch.tensor(1 / 6), torch.tensor(1 / 6)]

        # Also test that per-region aupros are averaged
        labels.append(torch.cat(labels))
        preds.append(torch.cat(preds))
        fpr_limit.append(float(np.mean(fpr_limit)))
        aupro.append(torch.tensor(np.mean(aupro)))

        thresholds = [
            torch.linspace(0, 1, steps=50),
            torch.linspace(0, 1, steps=50),
        ]
        vals = list(zip(labels, preds, thresholds, fpr_limit, aupro))

        metafunc.parametrize(argnames=("labels", "preds", "thresholds", "fpr_limit", "aupro"), argvalues=vals)


def test_pro(labels, preds, thresholds, fpr_limit, aupro):
    pro = AUPRO(fpr_limit=fpr_limit)
    pro.update(preds, labels)
    computed_aupro = pro.compute()

    tmp_labels = [label.squeeze().numpy() for label in labels]
    tmp_preds = [pred.squeeze().numpy() for pred in preds]
    ref_pro = torch.tensor(calculate_au_pro(tmp_labels, tmp_preds, integration_limit=fpr_limit)[0], dtype=torch.float)

    TOL = 0.001
    assert torch.allclose(computed_aupro, aupro, atol=TOL)
    assert torch.allclose(computed_aupro, ref_pro, atol=TOL)
    assert torch.allclose(aupro, ref_pro, atol=TOL)

    binned_pro = AUPRO(fpr_limit=fpr_limit, thresholds=thresholds)
    binned_pro.update(preds, labels)
    computed_binned_aupro = binned_pro.compute()

    assert computed_binned_aupro != computed_aupro
    assert torch.allclose(computed_aupro, aupro, atol=TOL)


