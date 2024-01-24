"""Test PRO metric."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torchvision.transforms import RandomAffine

from anomalib.data.utils import random_2d_perlin
from anomalib.metrics.pro import PRO, connected_components_cpu, connected_components_gpu


def test_pro() -> None:
    """Checks if PRO metric computes the (macro) average of the per-region overlap."""
    labels = torch.Tensor(
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
            ],
        ],
    )
    # ground truth mask is int type
    labels = labels.type(torch.int32)

    preds = (torch.arange(10) / 10) + 0.05
    # metrics receive squeezed predictions (N, H, W)
    preds = preds.unsqueeze(1).repeat(1, 5).view(1, 10, 5)

    thresholds = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    targets = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    for threshold, target in zip(thresholds, targets, strict=True):
        pro = PRO(threshold=threshold)
        pro.update(preds, labels)
        assert pro.compute() == target


def test_device_consistency() -> None:
    """Test if the pro metric yields the same results between cpu and gpu."""
    transform = RandomAffine(5, None, (0.95, 1.05), 5)

    batch = torch.zeros((32, 256, 256))
    for i in range(batch.shape[0]):
        batch[i, ...] = random_2d_perlin((256, 256), (torch.tensor(4), torch.tensor(4))) > 0.5
    # ground truth mask is int type
    batch = batch.type(torch.int32)

    preds = transform(batch)

    pro_cpu = PRO()
    pro_gpu = PRO()

    pro_cpu.update(preds.cpu(), batch.cpu())
    pro_gpu.update(preds.cuda(), batch.cuda())

    assert torch.isclose(pro_cpu.compute(), pro_gpu.compute().cpu())


def test_connected_component_labeling() -> None:
    """Tests if the connected component labeling algorithms on cpu and gpu yield the same result."""
    # generate batch of random binary images using perlin noise
    batch = torch.zeros((32, 1, 256, 256))
    for i in range(batch.shape[0]):
        batch[i, ...] = random_2d_perlin((256, 256), (torch.tensor(4), torch.tensor(4))) > 0.5

    # get connected component results on both cpu and gpu
    cc_cpu = connected_components_cpu(batch.cpu())
    cc_gpu = connected_components_gpu(batch.cuda())

    # check if comps are ordered from 0 to N
    assert len(cc_cpu.unique()) == cc_cpu.unique().max() + 1
    assert len(cc_gpu.unique()) == cc_gpu.unique().max() + 1
    # check if same number of comps found between cpu and gpu
    assert len(cc_cpu.unique()) == len(cc_gpu.unique())
