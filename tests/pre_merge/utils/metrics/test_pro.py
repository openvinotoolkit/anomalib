import torch
from torch import Tensor

from anomalib.utils.metrics.pro import PRO


def test_pro():
    """Checks if PRO metric computes the (macro) average of the per-region overlap."""

    labels = Tensor(
        [
            [
                [
                    [0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1],
                ]
            ]
        ]
    )

    preds = (torch.arange(10) / 10) + 0.05
    preds = preds.unsqueeze(1).repeat(1, 5).view(1, 1, 10, 5)

    thresholds = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    targets = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    for threshold, target in zip(thresholds, targets):
        pro = PRO(threshold=threshold)
        pro.update(preds, labels)
        assert pro.compute() == target
