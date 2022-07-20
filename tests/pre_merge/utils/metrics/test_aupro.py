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
                        [
                            [0, 0, 0, 1, 0, 0, 0],
                        ]
                        * 400,
                    ]
                ]
            ),
            torch.tensor(
                [
                    [
                        [
                            [0, 1, 0, 1, 0, 1, 0],
                        ]
                        * 400,
                    ]
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

        vals = list(zip(labels, preds, fpr_limit, aupro))
        metafunc.parametrize(argnames=("labels", "preds", "fpr_limit", "aupro"), argvalues=vals)


def test_pro(labels, preds, fpr_limit, aupro):
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
