# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GRD-Net ROI folder pipeline."""

import logging
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image as PILImage
from tests.helpers.data import DummyImageDatasetGenerator
from torchvision.transforms.v2 import Compose, RandomCrop, RandomRotation, Resize, Transform

from anomalib.data.utils import Split, ValSplitMode
from anomalib.models.image.grdnet.data.dataclasses import GRDNetBatch
from anomalib.models.image.grdnet.data.datamodule import GRDNetFolder
from anomalib.models.image.grdnet.data.dataset import GRDNetFolderDataset


def _write_image(path: Path, value: int = 64, shape: tuple[int, int] = (10, 12)) -> None:
    """Write a small RGB fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(np.full((*shape, 3), value, dtype=np.uint8)).save(path)


def _write_mask(path: Path, value: int = 255, shape: tuple[int, int] = (10, 12)) -> None:
    """Write a small binary mask fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(np.full(shape, value, dtype=np.uint8)).save(path)


@pytest.fixture
def folder_root(tmp_path: Path) -> Path:
    """Reuse the standard RGB/ground-truth generator and add only ROI fixtures."""
    generator = DummyImageDatasetGenerator(
        data_format="mvtecad",
        root=tmp_path,
        num_train=4,
        num_test=2,
        abnormal_category="bad",
        image_shape=(64, 64),
        seed=42,
    )
    generator.generate_dataset()
    root = generator.dataset_root / "dummy"
    for split in ("train", "test"):
        for path in (root / split).glob("*/*.png"):
            _write_mask(root / "roi" / path.relative_to(root), shape=(64, 64))
    return root


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


def test_absolute_roi_root(folder_root: Path, tmp_path: Path) -> None:
    """An absolute external ROI root mirrors paths relative to the dataset root."""
    roi_root = tmp_path / "external-roi"
    (folder_root / "roi").rename(roi_root)

    dataset = GRDNetFolderDataset(
        name="test",
        root=folder_root,
        normal_dir="train/good",
        roi_dir=roi_root,
    )

    assert Path(dataset[0].roi_mask_path).is_relative_to(roi_root)
    assert dataset[0].roi_mask.all()


@pytest.mark.parametrize("target_roi_exists", [False, True])
def test_image_symlink_preserves_sample_relative_roi(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    target_roi_exists: bool,
) -> None:
    """An in-root image symlink uses its sample path, not its target, for ROI lookup."""
    root = tmp_path / "dataset"
    _write_image(root / "pool/000.png")
    image_path = root / "train/good/000.png"
    image_path.parent.mkdir(parents=True)
    image_path.symlink_to(root / "pool/000.png")
    roi_path = root / "roi/train/good/000.png"
    _write_mask(roi_path, value=0)
    if target_roi_exists:
        _write_mask(root / "roi/pool/000.png")

    with caplog.at_level(logging.WARNING):
        dataset = GRDNetFolderDataset(name="test", root=root, normal_dir="train/good", roi_dir="roi")

    item = dataset[0]
    assert Path(item.roi_mask_path) == roi_path
    assert not item.roi_mask.any()
    assert not caplog.records


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


def test_no_roi_root_uses_full_mask_without_warning(folder_root: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Omitting ROI configuration silently uses full-image masks."""
    with caplog.at_level(logging.WARNING):
        dataset = GRDNetFolderDataset(name="test", root=folder_root, normal_dir="train/good")

    item = dataset[0]
    assert not caplog.records
    assert item.gt_mask is None
    assert item.roi_mask.all()


def test_nonexistent_roi_root_fails(folder_root: Path) -> None:
    """An explicitly configured missing ROI root is rejected."""
    with pytest.raises(FileNotFoundError):
        GRDNetFolderDataset(name="test", root=folder_root, normal_dir="train/good", roi_dir="missing")


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

    image_path = root / "train/good/000.png"
    image_path.unlink()
    image_path.symlink_to(outside / "000.png")
    with pytest.raises(ValueError, match="outside the allowed directory"):
        GRDNetFolderDataset(name="test", root=root, normal_dir="train/good")


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


def test_datamodule_collates_grdnet_batches(folder_root: Path) -> None:
    """GRD-Net folder splits preserve their custom batch type."""
    datamodule = GRDNetFolder(
        name="test",
        root=folder_root,
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


@pytest.mark.parametrize("escape", [False, True])
def test_ground_truth_paths_are_confined(folder_root: Path, tmp_path: Path, escape: bool) -> None:
    """Ground-truth reads permit in-root files but reject escaping symlinks."""
    dataset = GRDNetFolderDataset(
        name="test",
        root=folder_root,
        normal_dir="train/good",
        abnormal_dir="test/bad",
        mask_dir="ground_truth/bad",
        split=Split.TEST,
    )
    if escape:
        mask_path = Path(dataset.samples.iloc[0].mask_path)
        outside = tmp_path / "outside-mask.png"
        _write_mask(outside, shape=(64, 64))
        mask_path.unlink()
        mask_path.symlink_to(outside)
        with pytest.raises(ValueError, match="outside the allowed directory"):
            dataset[0]
    else:
        item = dataset[0]
        assert item.gt_mask is not None
        assert item.roi_mask.all()
