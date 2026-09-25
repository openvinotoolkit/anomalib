# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GRD-Net ROI folder pipeline."""

import logging
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image as PILImage
from torchvision.transforms.v2 import Compose, RandomCrop, RandomRotation, Resize, Transform

from anomalib.data.dataclasses.torch.grdnet import GRDNetBatch
from anomalib.data.datamodules.image.grdnet import GRDNetFolder
from anomalib.data.datasets.image.grdnet import GRDNetFolderDataset
from anomalib.data.utils import Split, ValSplitMode


def _write_image(path: Path, value: int = 64, shape: tuple[int, int] = (10, 12)) -> None:
    """Write a small RGB fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(np.full((*shape, 3), value, dtype=np.uint8)).save(path)


def _write_mask(path: Path, value: int = 255, shape: tuple[int, int] = (10, 12)) -> None:
    """Write a small binary mask fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(np.full(shape, value, dtype=np.uint8)).save(path)


def _make_folder_tree(root: Path) -> None:
    """Create a minimal folder dataset with train and test data."""
    for index in range(2):
        _write_image(root / "train/good" / f"{index:03}.png", value=32 + index)
        _write_mask(root / "roi/train/good" / f"{index:03}.png")
    _write_image(root / "test/good/000.png")
    _write_mask(root / "roi/test/good/000.png")
    _write_image(root / "test/bad/000.png", value=192)
    _write_mask(root / "ground_truth/bad/000.png")
    _write_mask(root / "roi/test/bad/000.png")


def test_roi_resolution_prefers_exact_suffix_and_supports_png(tmp_path: Path) -> None:
    """Exact-suffix masks win before the PNG fallback for relative ROI roots."""
    root = tmp_path / "dataset"
    _write_image(root / "train/good/000.jpg")
    _write_image(root / "train/good/001.jpg")
    _write_mask(root / "roi/train/good/000.jpg", value=0)
    _write_mask(root / "roi/train/good/000.png", value=255)
    _write_mask(root / "roi/train/good/001.png", value=255)

    dataset = GRDNetFolderDataset(
        name="test",
        root=root,
        normal_dir="train/good",
        roi_dir="roi",
        split=Split.TRAIN,
    )

    assert Path(dataset.samples.iloc[0].roi_mask_path).suffix == ".jpg"
    assert not dataset[0].roi_mask.any()
    assert Path(dataset.samples.iloc[1].roi_mask_path).suffix == ".png"
    assert dataset[1].roi_mask.all()


def test_absolute_roi_root(tmp_path: Path) -> None:
    """An absolute external ROI root mirrors paths relative to the dataset root."""
    root = tmp_path / "dataset"
    roi_root = tmp_path / "external-roi"
    _write_image(root / "train/good/000.png")
    _write_mask(roi_root / "train/good/000.png")

    dataset = GRDNetFolderDataset(
        name="test",
        root=root,
        normal_dir="train/good",
        roi_dir=roi_root,
    )

    assert Path(dataset[0].roi_mask_path).is_relative_to(roi_root)
    assert dataset[0].roi_mask.all()


def test_missing_roi_uses_native_full_image_mask(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Missing masks emit one summary and use a native-size full-image ROI."""
    root = tmp_path / "dataset"
    _write_image(root / "train/good/000.png", shape=(7, 11))
    (root / "roi").mkdir(parents=True)

    with caplog.at_level(logging.WARNING):
        dataset = GRDNetFolderDataset(
            name="test",
            root=root,
            normal_dir="train/good",
            roi_dir="roi",
        )

    assert len(caplog.records) == 1
    assert "1 of 1 images" in caplog.text
    assert dataset[0].roi_mask_path == ""
    assert dataset[0].roi_mask.shape == (7, 11)
    assert dataset[0].roi_mask.all()


def test_no_roi_root_uses_full_mask_without_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Omitting ROI configuration silently uses full-image masks."""
    root = tmp_path / "dataset"
    _write_image(root / "train/good/000.png")

    with caplog.at_level(logging.WARNING):
        dataset = GRDNetFolderDataset(name="test", root=root, normal_dir="train/good")

    assert not caplog.records
    assert dataset[0].roi_mask.all()


def test_nonexistent_roi_root_fails(tmp_path: Path) -> None:
    """An explicitly configured missing ROI root is rejected."""
    root = tmp_path / "dataset"
    _write_image(root / "train/good/000.png")

    with pytest.raises(FileNotFoundError):
        GRDNetFolderDataset(name="test", root=root, normal_dir="train/good", roi_dir="missing")


def test_image_and_roi_paths_are_confined(tmp_path: Path) -> None:
    """Images and ROI symlinks cannot escape their configured roots."""
    root = tmp_path / "dataset"
    outside = tmp_path / "outside"
    root.mkdir()
    _write_image(outside / "000.png")
    with pytest.raises(ValueError, match="outside the allowed directory"):
        GRDNetFolderDataset(name="test", root=root, normal_dir=outside)

    _write_image(root / "train/good/000.png")
    _write_mask(outside / "roi.png")
    roi_link = root / "roi/train/good/000.png"
    roi_link.parent.mkdir(parents=True)
    roi_link.symlink_to(outside / "roi.png")
    with pytest.raises(ValueError, match="outside the allowed directory"):
        GRDNetFolderDataset(name="test", root=root, normal_dir="train/good", roi_dir="roi")


class _CountingTransform(Transform):
    """Count calls while applying deterministic joint geometry."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.input_count = 0
        self.mask_values: torch.Tensor | None = None
        self.pipeline = Compose([Resize((16, 16)), RandomCrop((12, 12)), RandomRotation((45, 45))])

    def forward(self, *inputs: object) -> object:
        """Apply one joint transform call."""
        self.calls += 1
        self.input_count = len(inputs)
        outputs = self.pipeline(*inputs)
        self.mask_values = torch.unique(outputs[1])
        return outputs


def test_joint_augmentations_keep_masks_aligned(tmp_path: Path) -> None:
    """Image, anomaly mask, and ROI use one joint torchvision transform call."""
    root = tmp_path / "dataset"
    _write_image(root / "train/good/000.png")
    _write_image(root / "test/bad/000.png")
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2:8, 4:9] = 255
    for path in (root / "ground_truth/bad/000.png", root / "roi/test/bad/000.png"):
        path.parent.mkdir(parents=True, exist_ok=True)
        PILImage.fromarray(mask).save(path)
    transform = _CountingTransform()
    dataset = GRDNetFolderDataset(
        name="test",
        root=root,
        normal_dir="train/good",
        abnormal_dir="test/bad",
        mask_dir="ground_truth/bad",
        roi_dir="roi",
        split=Split.TEST,
        augmentations=transform,
    )

    item = dataset[0]

    assert transform.calls == 1
    assert transform.input_count == 3
    assert transform.mask_values is not None
    assert set(transform.mask_values.tolist()).issubset({0, 1})
    assert item.image.shape == (3, 12, 12)
    assert item.gt_mask.dtype == torch.bool
    assert item.roi_mask.dtype == torch.bool
    assert torch.equal(item.gt_mask, item.roi_mask)


def test_classification_and_segmentation_samples(tmp_path: Path) -> None:
    """Both folder task types return GRD-Net items with materialized ROI masks."""
    classification_root = tmp_path / "classification"
    _write_image(classification_root / "good/000.png")
    classification = GRDNetFolderDataset(name="classification", root=classification_root, normal_dir="good")

    segmentation_root = tmp_path / "segmentation"
    _write_image(segmentation_root / "train/good/000.png")
    _write_image(segmentation_root / "test/bad/000.png")
    _write_mask(segmentation_root / "ground_truth/bad/000.png")
    segmentation = GRDNetFolderDataset(
        name="segmentation",
        root=segmentation_root,
        normal_dir="train/good",
        abnormal_dir="test/bad",
        mask_dir="ground_truth/bad",
        split=Split.TEST,
    )

    assert classification[0].gt_mask is None
    assert classification[0].roi_mask.all()
    assert segmentation[0].gt_mask is not None
    assert segmentation[0].roi_mask.all()


def test_datamodule_collates_grdnet_batches(tmp_path: Path) -> None:
    """GRD-Net folder splits preserve their custom batch type."""
    root = tmp_path / "dataset"
    _make_folder_tree(root)
    datamodule = GRDNetFolder(
        name="test",
        root=root,
        normal_dir="train/good",
        abnormal_dir="test/bad",
        normal_test_dir="test/good",
        mask_dir="ground_truth/bad",
        roi_dir="roi",
        train_batch_size=2,
        eval_batch_size=2,
        num_workers=0,
        val_split_mode=ValSplitMode.SAME_AS_TEST,
    )
    datamodule.setup()

    for dataloader in (datamodule.train_dataloader(), datamodule.val_dataloader(), datamodule.test_dataloader()):
        batch = next(iter(dataloader))
        assert isinstance(batch, GRDNetBatch)
        assert batch.image.shape[-2:] == batch.roi_mask.shape[-2:]
        assert batch.roi_mask_path is not None
