"""Tests for the collating DatasetItems into Batches."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from torchvision.tv_tensors import Image, Mask

from anomalib.data.dataclasses.generic import BatchIterateMixin


@dataclass
class DummyDatasetItem:
    """Dummy dataset item with image and mask."""

    image: Image
    mask: Mask


@dataclass
class DummyBatch(BatchIterateMixin[DummyDatasetItem]):
    """Dummy batch with image and mask."""

    item_class = DummyDatasetItem
    image: Image
    mask: Mask


def test_collate_heterogeneous_shapes() -> None:
    """Test collating items with different shapes."""
    items = [
        DummyDatasetItem(
            image=Image(torch.rand((3, 256, 256))),
            mask=Mask(torch.ones((256, 256))),
        ),
        DummyDatasetItem(
            image=Image(torch.rand((3, 224, 224))),
            mask=Mask(torch.ones((224, 224))),
        ),
    ]
    batch = DummyBatch.collate(items)
    # the collated batch should have the shape of the largest item
    assert batch.image.shape == (2, 3, 256, 256)
