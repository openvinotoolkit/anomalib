# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for GRD-Net joint preprocessing."""

import pytest
import torch

from anomalib.data import ImageBatch
from anomalib.data.dataclasses.torch.grdnet import GRDNetBatch
from anomalib.models.image.grdnet.pre_processor import GRDNetPreProcessor


@pytest.mark.parametrize(
    "hook_name",
    [
        "on_train_batch_start",
        "on_validation_batch_start",
        "on_test_batch_start",
        "on_predict_batch_start",
    ],
)
def test_callbacks_resize_image_and_masks_together(hook_name: str) -> None:
    """Every runtime callback applies the canonical geometry to all spatial fields."""
    roi_mask = torch.zeros(1, 9, 11)
    roi_mask[:, 2:7, 3:8] = 1
    batch = GRDNetBatch(
        image=roi_mask.unsqueeze(1).repeat(1, 3, 1, 1),
        gt_mask=roi_mask.clone(),
        roi_mask=roi_mask.clone(),
        roi_mask_path=[""],
    )
    pre_processor = GRDNetPreProcessor()

    getattr(pre_processor, hook_name)(None, None, batch, 0)

    assert batch.image.shape == (1, 3, 256, 256)
    assert batch.gt_mask.shape == (1, 256, 256)
    assert batch.roi_mask.shape == (1, 256, 256)
    assert batch.gt_mask.dtype == torch.bool
    assert batch.roi_mask.dtype == torch.bool
    assert torch.equal(batch.gt_mask, batch.roi_mask)


def test_standard_image_batch_is_supported() -> None:
    """The preprocessor retains compatibility with ordinary anomalib image batches."""
    batch = ImageBatch(image=torch.rand(1, 3, 9, 11), gt_mask=None)
    pre_processor = GRDNetPreProcessor()

    pre_processor.on_train_batch_start(None, None, batch, 0)

    assert batch.image.shape == (1, 3, 256, 256)
    assert batch.gt_mask is None


def test_grdnet_classification_batch_is_supported() -> None:
    """ROI-aware classification batches do not require anomaly ground-truth masks."""
    batch = GRDNetBatch(
        image=torch.rand(1, 3, 9, 11),
        gt_mask=None,
        roi_mask=torch.ones(1, 9, 11),
        roi_mask_path=[""],
    )
    pre_processor = GRDNetPreProcessor()

    pre_processor.on_train_batch_start(None, None, batch, 0)

    assert batch.image.shape == (1, 3, 256, 256)
    assert batch.gt_mask is None
    assert batch.roi_mask.shape == (1, 256, 256)


def test_export_transform_resizes_image_only() -> None:
    """The inherited forward path remains image-only and exportable."""
    pre_processor = GRDNetPreProcessor()
    output = pre_processor(torch.rand(1, 3, 9, 11))

    assert output.shape == (1, 3, 256, 256)
