"""Tests for the AUPRO metric."""

import numpy as np
import torch

from anomalib.utils.metrics import AUPRO


def pytest_generate_tests(metafunc):
    if metafunc.function == test_pro:
        labels = [
            torch.tensor(
                [
                    [
                        [
                            [0, 0, 0, 1, 0, 0, 0],
                        ]
                        * 10,
                    ]
                ]
            ),
            torch.tensor(
                [
                    [
                        [
                            [0, 0, 0, 0, 0, 1, 0],
                        ]
                        * 10,
                    ]
                ]
            ),
            torch.tensor(
                [
                    [
                        [
                            [0, 0, 0, 0, 1, 1, 0],
                        ]
                        * 10,
                    ]
                ]
            ),
        ]
        preds = torch.arange(70) / 70.0
        preds = preds.view(1, 1, 10, 7)

        preds = [preds, preds, preds]

        fpr_limit = [1 / 3, 1 / 3, 1 / 3]
        aupro = [0.165, 0.2, 0.196]

        # Also test that per-region aupros are averaged
        labels.append(torch.cat(labels))
        preds.append(torch.cat(preds))
        fpr_limit.append(float(np.mean(fpr_limit)))
        aupro.append(float(np.mean(aupro)))

        vals = list(zip(labels, preds, fpr_limit, aupro))
        metafunc.parametrize(argnames=("labels", "preds", "fpr_limit", "aupro"), argvalues=vals)


def test_pro(labels, preds, fpr_limit, aupro):
    pro = AUPRO(fpr_limit=fpr_limit)
    pro.update(preds, labels)
    computed_aupro = pro.compute()

    assert torch.allclose(
        computed_aupro, torch.tensor(aupro), atol=0.001
    )  # Need high atol due to dumensionality of the problem
