import torch
, all as torch_all

from anomalib.utils.metrics.binning import thresholds_between_min_and_max, thresholds_between_0_and_1


def test_thresholds_between_min_and_max():
    preds = torch.Tensor([1, 10])
    assert torch.all(thresholds_between_min_and_max(preds, 2) == preds)


def test_thresholds_between_0_and_1():
    expected = torch.Tensor([0, 1])
    assert torch.all(thresholds_between_0_and_1(2) == expected)
