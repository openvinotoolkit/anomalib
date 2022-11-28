"""Tests the custom collate function used in Anomalib."""

from typing import List

import pytest
import torch
from torch import Tensor

from anomalib.data.base.datamodule import collate_fn


@pytest.fixture
def boxes_entry():
    return {"boxes": torch.rand((6, 4))}


@pytest.fixture
def other_entry():
    return {"other": torch.rand((256, 256))}


@pytest.mark.parametrize("batch_size", [1, 2, 8])
@pytest.mark.parametrize(
    ("boxes_present", "other_present"), [(True, True), (True, False), (False, True), (False, False)]
)
def test_collate_fn(batch_size, boxes_present, other_present, boxes_entry, other_entry):
    """Tests the custom collate function used by the anomalib dataloaders"""

    batch = []
    for _ in range(batch_size):
        elem = {}
        if boxes_present:
            elem.update(boxes_entry)
        if other_present:
            elem.update(other_entry)
        batch.append(elem)

    collated_batch = collate_fn(batch)
    if boxes_present:
        assert isinstance(collated_batch["boxes"], List)
        assert isinstance(collated_batch["boxes"][0], Tensor)
    if other_present:
        assert isinstance(collated_batch["other"], Tensor)
