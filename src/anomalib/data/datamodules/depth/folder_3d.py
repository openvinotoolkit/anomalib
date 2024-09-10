"""Custom Folder Datamodule.

This script creates a custom datamodule from a folder.
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from torchvision.transforms.v2 import Transform

from anomalib import TaskType
from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.depth.folder_3d import Folder3DDataset
from anomalib.data.utils import Split, TestSplitMode, ValSplitMode


class Folder3D(AnomalibDataModule):
    """Folder DataModule.

    Args:
        name (str): Name of the dataset. This is used to name the datamodule, especially when logging/saving.
        normal_dir (str | Path): Name of the directory containing normal images.
        root (str | Path | None): Path to the root folder containing normal and abnormal dirs.
            Defaults to ``None``.
        abnormal_dir (str | Path | None): Name of the directory containing abnormal images.
            Defaults to ``abnormal``.
        normal_test_dir (str | Path | None, optional): Path to the directory containing normal images for the test
            dataset.
            Defaults to ``None``.
        mask_dir (str | Path | None, optional): Path to the directory containing the mask annotations.
            Defaults to ``None``.
        normal_depth_dir (str | Path | None, optional): Path to the directory containing
            normal depth images for the test dataset. Normal test depth images will be a split of `normal_dir`
        abnormal_depth_dir (str | Path | None, optional): Path to the directory containing
            abnormal depth images for the test dataset.
        normal_test_depth_dir (str | Path | None, optional): Path to the directory containing
            normal depth images for the test dataset. Normal test images will be a split of `normal_dir`
            if `None`. Defaults to None.
        normal_split_ratio (float, optional): Ratio to split normal training images and add to the
            test set in case test set doesn't contain any normal images.
            Defaults to 0.2.
        extensions (tuple[str, ...] | None, optional): Type of the image extensions to read from the
            directory. Defaults to None.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Test batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            Defaults to ``8``.
        task (TaskType, optional): Task type. Could be ``classification``, ``detection`` or ``segmentation``.
            Defaults to ``TaskType.SEGMENTATION``.
        image_size (tuple[int, int], optional): Size to which input images should be resized.
            Defaults to ``None``.
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
        train_transform (Transform, optional): Transforms that should be applied to the input images during training.
            Defaults to ``None``.
        eval_transform (Transform, optional): Transforms that should be applied to the input images during evaluation.
            Defaults to ``None``.
        test_split_mode (TestSplitMode): Setting that determines how the testing subset is obtained.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float): Fraction of images from the train set that will be reserved for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
            Defaults to ``ValSplitMode.FROM_TEST``.
        val_split_ratio (float): Fraction of train or test images that will be reserved for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Seed used during random subset splitting.
            Defaults to ``None``.
    """

    def __init__(
        self,
        name: str,
        normal_dir: str | Path,
        root: str | Path,
        abnormal_dir: str | Path | None = None,
        normal_test_dir: str | Path | None = None,
        mask_dir: str | Path | None = None,
        normal_depth_dir: str | Path | None = None,
        abnormal_depth_dir: str | Path | None = None,
        normal_test_depth_dir: str | Path | None = None,
        extensions: tuple[str] | None = None,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        task: TaskType | str = TaskType.SEGMENTATION,
        image_size: tuple[int, int] | None = None,
        transform: Transform | None = None,
        train_transform: Transform | None = None,
        eval_transform: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode | str = ValSplitMode.FROM_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            image_size=image_size,
            transform=transform,
            train_transform=train_transform,
            eval_transform=eval_transform,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )
        self._name = name
        self.task = TaskType(task)
        self.root = Path(root)
        self.normal_dir = normal_dir
        self.abnormal_dir = abnormal_dir
        self.normal_test_dir = normal_test_dir
        self.mask_dir = mask_dir
        self.normal_depth_dir = normal_depth_dir
        self.abnormal_depth_dir = abnormal_depth_dir
        self.normal_test_depth_dir = normal_test_depth_dir
        self.extensions = extensions

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = Folder3DDataset(
            name=self.name,
            task=self.task,
            transform=self.train_transform,
            split=Split.TRAIN,
            root=self.root,
            normal_dir=self.normal_dir,
            abnormal_dir=self.abnormal_dir,
            normal_test_dir=self.normal_test_dir,
            mask_dir=self.mask_dir,
            normal_depth_dir=self.normal_depth_dir,
            abnormal_depth_dir=self.abnormal_depth_dir,
            normal_test_depth_dir=self.normal_test_depth_dir,
            extensions=self.extensions,
        )

        self.test_data = Folder3DDataset(
            name=self.name,
            task=self.task,
            transform=self.eval_transform,
            split=Split.TEST,
            root=self.root,
            normal_dir=self.normal_dir,
            abnormal_dir=self.abnormal_dir,
            normal_test_dir=self.normal_test_dir,
            normal_depth_dir=self.normal_depth_dir,
            abnormal_depth_dir=self.abnormal_depth_dir,
            normal_test_depth_dir=self.normal_test_depth_dir,
            mask_dir=self.mask_dir,
            extensions=self.extensions,
        )

    @property
    def name(self) -> str:
        """Name of the datamodule.

        Folder3D datamodule overrides the name property to provide a custom name.
        """
        return self._name
