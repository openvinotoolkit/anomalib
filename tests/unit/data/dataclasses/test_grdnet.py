# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for GRD-Net ROI dataclasses."""

import pytest
import torch
from lightning.fabric.utilities.apply_func import move_data_to_device
from torchvision.tv_tensors import Mask

from anomalib.data.dataclasses.numpy.image import NumpyImageBatch, NumpyImageItem
from anomalib.data.dataclasses.torch.grdnet import GRDNetBatch, GRDNetItem


def _item(index: int = 0) -> GRDNetItem:
    """Create a deterministic GRD-Net item."""
    return GRDNetItem(
        image=torch.full((3, 8, 10), index / 10),
        gt_label=index % 2,
        gt_mask=torch.full((8, 10), index % 2),
        image_path=f"image-{index}.png",
        mask_path="",
        roi_mask=torch.ones(1, 8, 10),
        roi_mask_path="" if index == 0 else f"roi-{index}.png",
    )


def test_item_validates_roi_fields() -> None:
    """ROI item fields are normalized without losing fallback paths."""
    item = _item()

    assert isinstance(item.roi_mask, Mask)
    assert item.roi_mask.dtype == torch.bool
    assert item.roi_mask.shape == (8, 10)
    assert item.roi_mask_path == ""


@pytest.mark.parametrize("shape", [(2, 8, 10), (1, 1, 8, 10)])
def test_item_rejects_invalid_roi_shape(shape: tuple[int, ...]) -> None:
    """An item accepts only one two-dimensional ROI mask."""
    with pytest.raises(ValueError, match="ROI mask must"):
        GRDNetItem(image=torch.rand(3, 8, 10), roi_mask=torch.ones(shape))


def test_item_rejects_spatial_mismatch() -> None:
    """Image and ROI spatial dimensions must agree."""
    with pytest.raises(ValueError, match="spatial dimensions must match"):
        GRDNetItem(image=torch.rand(3, 8, 10), roi_mask=torch.ones(7, 10))


def test_collate_iterate_and_update() -> None:
    """ROI fields survive collation, iteration, and dataclass updates."""
    batch = GRDNetBatch.collate([_item(0), _item(1)])

    assert isinstance(batch, GRDNetBatch)
    assert batch.roi_mask.shape == (2, 8, 10)
    assert batch.roi_mask.dtype == torch.bool
    assert batch.roi_mask_path == ["", "roi-1.png"]
    assert [item.roi_mask_path for item in batch] == ["", "roi-1.png"]

    replacement = torch.zeros(2, 8, 10)
    updated = batch.update(in_place=False, roi_mask=replacement)
    assert updated is not batch
    assert not updated.roi_mask.any()
    assert batch.roi_mask.all()


def test_batch_validates_size_and_paths() -> None:
    """ROI batch and path counts must match the image batch."""
    with pytest.raises(ValueError, match="batch/spatial dimensions must match"):
        GRDNetBatch(image=torch.rand(2, 3, 8, 10), roi_mask=torch.ones(1, 8, 10))
    with pytest.raises(ValueError, match="does not match the image batch size"):
        GRDNetBatch(image=torch.rand(2, 3, 8, 10), roi_mask_path=[""])
    with pytest.raises(TypeError, match="not a single string"):
        GRDNetBatch(image=torch.rand(1, 3, 8, 10), roi_mask_path="")


def test_device_transfer_includes_roi_mask() -> None:
    """Lightning device transfer moves the ROI mask with the shared fields."""
    batch = GRDNetBatch.collate([_item()])
    moved = move_data_to_device(batch, torch.device("meta"))

    assert moved.image.device.type == "meta"
    assert moved.roi_mask.device.type == "meta"
    assert moved.roi_mask_path == [""]


def test_numpy_projection_omits_roi_fields() -> None:
    """NumPy projection remains compatible with standard image consumers."""
    item = _item().to_numpy()
    grdnet_batch = GRDNetBatch.collate([_item(0), _item(1)])
    grdnet_batch.update(
        pred_score=torch.tensor([0.1, 0.2]),
        anomaly_map=torch.rand(2, 8, 10),
    )
    batch = grdnet_batch.to_numpy()

    assert isinstance(item, NumpyImageItem)
    assert isinstance(batch, NumpyImageBatch)
    assert batch.pred_score.shape == (2,)
    assert batch.anomaly_map.shape == (2, 8, 10)
    assert not hasattr(item, "roi_mask")
    assert not hasattr(batch, "roi_mask")
