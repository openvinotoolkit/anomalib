# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Folder dataset with region-of-interest supervision for GRD-Net."""

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from torchvision.transforms.v2 import Transform
from torchvision.tv_tensors import Mask

from anomalib import TaskType
from anomalib.data.dataclasses.torch.grdnet import GRDNetBatch, GRDNetItem
from anomalib.data.datasets.image.folder import FolderDataset
from anomalib.data.utils import LabelName, Split, read_image, read_mask
from anomalib.data.utils.path import resolve_path_under_root, validate_path

logger = logging.getLogger(__name__)


class GRDNetFolderDataset(FolderDataset):
    """Load folder data with optional ROI masks for GRD-Net training.

    ROI paths mirror image paths relative to ``root``. For each image, a mask with the same suffix is preferred,
    followed by a ``.png`` mask. Missing masks are represented by a full-image ROI.

    Args:
        name: Name of the dataset.
        normal_dir: Directory or directories containing normal images.
        root: Root directory containing every source image.
        roi_dir: ROI directory relative to ``root`` or an absolute external directory.
        augmentations: Joint torchvision v2 augmentations.
        abnormal_dir: Directory or directories containing anomalous images.
        normal_test_dir: Directory or directories containing normal test images.
        mask_dir: Directory or directories containing anomaly ground-truth masks.
        split: Dataset split to load.
        extensions: Image file extensions to include.
    """

    def __init__(
        self,
        name: str,
        normal_dir: str | Path | Sequence[str | Path],
        root: str | Path,
        roi_dir: str | Path | None = None,
        augmentations: Transform | None = None,
        abnormal_dir: str | Path | Sequence[str | Path] | None = None,
        normal_test_dir: str | Path | Sequence[str | Path] | None = None,
        mask_dir: str | Path | Sequence[str | Path] | None = None,
        split: str | Split | None = None,
        extensions: tuple[str, ...] | None = None,
    ) -> None:
        dataset_root = validate_path(root)
        if not dataset_root.is_dir():
            msg = f"Dataset root must be a directory, got {dataset_root}."
            raise NotADirectoryError(msg)

        super().__init__(
            name=name,
            normal_dir=normal_dir,
            augmentations=augmentations,
            root=dataset_root,
            abnormal_dir=abnormal_dir,
            normal_test_dir=normal_test_dir,
            mask_dir=mask_dir,
            split=split,
            extensions=extensions,
        )
        self.dataset_root = dataset_root
        self.roi_dir = _resolve_roi_root(dataset_root, roi_dir)
        self._add_roi_paths()

    def _add_roi_paths(self) -> None:
        """Resolve and attach one confined ROI path to every sample."""
        samples = self.samples.copy()
        samples.attrs = self.samples.attrs.copy()
        roi_paths = []
        missing = 0
        for image_path in samples["image_path"]:
            image = validate_path(image_path, base_dir=self.dataset_root)
            roi_path = _find_roi_path(image, self.dataset_root, self.roi_dir)
            roi_paths.append(str(roi_path) if roi_path is not None else "")
            missing += roi_path is None
        samples["roi_mask_path"] = roi_paths
        self.samples = samples

        if self.roi_dir is not None and missing:
            logger.warning(
                "ROI masks were not found for %d of %d images in %s; full-image ROI masks will be used.",
                missing,
                len(samples),
                self.roi_dir,
            )

    def __getitem__(self, index: int) -> GRDNetItem:
        """Load one image, anomaly mask, and spatially aligned ROI mask.

        Args:
            index: Dataset index.

        Returns:
            GRD-Net item containing the image and ROI supervision.
        """
        sample = self.samples.iloc[index]
        image = read_image(sample.image_path, as_tensor=True)

        gt_mask = None
        if self.task == TaskType.SEGMENTATION:
            if sample.label_index == LabelName.NORMAL:
                gt_mask = Mask(torch.zeros(image.shape[-2:], dtype=torch.uint8))
            elif sample.label_index == LabelName.ABNORMAL:
                gt_mask = read_mask(sample.mask_path, as_tensor=True)

        roi_mask = (
            read_mask(sample.roi_mask_path, as_tensor=True)
            if sample.roi_mask_path
            else Mask(torch.ones(image.shape[-2:], dtype=torch.uint8))
        )

        if self.augmentations:
            temporary_gt_mask = (
                gt_mask if gt_mask is not None else Mask(torch.zeros(image.shape[-2:], dtype=torch.uint8))
            )
            image, augmented_gt_mask, roi_mask = self.augmentations(image, temporary_gt_mask, roi_mask)
            if gt_mask is not None:
                gt_mask = augmented_gt_mask

        gt_label = None if sample.label_index == LabelName.UNKNOWN else torch.tensor(sample.label_index)
        return GRDNetItem(
            image=image,
            gt_mask=gt_mask,
            gt_label=gt_label,
            image_path=sample.image_path,
            mask_path=sample.mask_path,
            roi_mask=roi_mask,
            roi_mask_path=sample.roi_mask_path,
        )

    @property
    def collate_fn(self) -> Callable:
        """Return the GRD-Net batch collation function.

        Returns:
            GRD-Net batch collation function.
        """
        return GRDNetBatch.collate


def _resolve_roi_root(dataset_root: Path, roi_dir: str | Path | None) -> Path | None:
    """Resolve the optional ROI root while preserving its supported external form."""
    if roi_dir is None:
        return None
    roi_path = Path(roi_dir)
    resolved = validate_path(roi_path) if roi_path.is_absolute() else resolve_path_under_root(dataset_root, roi_path)
    if not resolved.is_dir():
        msg = f"ROI root must be a directory, got {resolved}."
        raise NotADirectoryError(msg)
    return resolved


def _find_roi_path(image_path: Path, dataset_root: Path, roi_root: Path | None) -> Path | None:
    """Find a confined ROI mask that mirrors an image path."""
    if roi_root is None:
        return None

    relative_path = image_path.resolve().relative_to(dataset_root.resolve())
    candidates = [relative_path]
    png_path = relative_path.with_suffix(".png")
    if png_path != relative_path:
        candidates.append(png_path)

    for candidate in candidates:
        resolved = resolve_path_under_root(roi_root, candidate, should_exist=False)
        if resolved.is_file():
            return resolved
    return None
