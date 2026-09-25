# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Folder datamodule with region-of-interest supervision for GRD-Net."""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.image.folder import Folder
from anomalib.data.datasets.image.grdnet import GRDNetFolderDataset
from anomalib.data.utils import Split, TestSplitMode, ValSplitMode


class GRDNetFolder(Folder):
    """Load a folder dataset with optional training ROI masks.

    Args:
        name: Name of the dataset.
        normal_dir: Directory or directories containing normal images.
        root: Root directory containing every source image.
        roi_dir: ROI directory relative to ``root`` or an absolute external directory.
        abnormal_dir: Directory or directories containing anomalous images.
        normal_test_dir: Directory or directories containing normal test images.
        mask_dir: Directory or directories containing anomaly ground-truth masks.
        normal_split_ratio: Ratio used to split normal images when needed.
        extensions: Image file extensions to include.
        train_batch_size: Training batch size.
        eval_batch_size: Validation and test batch size.
        num_workers: Number of data loading workers.
        train_augmentations: Training augmentations.
        val_augmentations: Validation augmentations.
        test_augmentations: Test augmentations.
        augmentations: Augmentations used when a stage-specific transform is absent.
        test_split_mode: Method used to construct the test split.
        test_split_ratio: Fraction used when splitting test data.
        val_split_mode: Method used to construct the validation split.
        val_split_ratio: Fraction used when splitting validation data.
        seed: Random seed used for dataset splitting.
    """

    def __init__(
        self,
        name: str,
        normal_dir: str | Path | Sequence[str | Path],
        root: str | Path,
        roi_dir: str | Path | None = None,
        abnormal_dir: str | Path | Sequence[str | Path] | None = None,
        normal_test_dir: str | Path | Sequence[str | Path] | None = None,
        mask_dir: str | Path | Sequence[str | Path] | None = None,
        normal_split_ratio: float = 0.2,
        extensions: tuple[str, ...] | None = None,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode | str = ValSplitMode.FROM_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            name=name,
            normal_dir=normal_dir,
            root=root,
            abnormal_dir=abnormal_dir,
            normal_test_dir=normal_test_dir,
            mask_dir=mask_dir,
            normal_split_ratio=normal_split_ratio,
            extensions=cast("tuple[str] | None", extensions),
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            train_augmentations=train_augmentations,
            val_augmentations=val_augmentations,
            test_augmentations=test_augmentations,
            augmentations=augmentations,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )
        self.roi_dir = roi_dir

    def _setup(self, _stage: str | None = None) -> None:
        """Create the train and test GRD-Net folder datasets."""
        if self.root is None:
            msg = "GRDNetFolder requires a dataset root."
            raise RuntimeError(msg)
        self.train_data = GRDNetFolderDataset(
            name=self.name,
            root=self.root,
            roi_dir=self.roi_dir,
            normal_dir=self.normal_dir,
            abnormal_dir=self.abnormal_dir,
            normal_test_dir=self.normal_test_dir,
            mask_dir=self.mask_dir,
            extensions=self.extensions,
            split=Split.TRAIN,
        )
        self.test_data = GRDNetFolderDataset(
            name=self.name,
            root=self.root,
            roi_dir=self.roi_dir,
            normal_dir=self.normal_dir,
            abnormal_dir=self.abnormal_dir,
            normal_test_dir=self.normal_test_dir,
            mask_dir=self.mask_dir,
            extensions=self.extensions,
            split=Split.TEST,
        )
