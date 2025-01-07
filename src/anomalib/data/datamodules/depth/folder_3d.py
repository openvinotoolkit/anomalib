"""Custom Folder Datamodule for 3D data.

This module provides a custom datamodule for handling 3D data organized in folders.
The datamodule supports RGB and depth image pairs for anomaly detection tasks.

Example:
    Create a folder 3D datamodule::

        >>> from anomalib.data import Folder3D
        >>> datamodule = Folder3D(
        ...     name="my_dataset",
        ...     root="path/to/dataset",
        ...     normal_dir="normal",
        ...     abnormal_dir="abnormal"
        ... )
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.depth.folder_3d import Folder3DDataset
from anomalib.data.utils import Split, TestSplitMode, ValSplitMode


class Folder3D(AnomalibDataModule):
    """Folder DataModule for 3D data.

    This class extends :class:`AnomalibDataModule` to handle datasets containing
    RGB and depth image pairs organized in folders.

    Args:
        name (str): Name of the dataset used for logging and saving.
        normal_dir (str | Path): Directory containing normal RGB images.
        root (str | Path): Root folder containing normal and abnormal dirs.
        abnormal_dir (str | Path | None, optional): Directory containing abnormal
            RGB images. Defaults to ``None``.
        normal_test_dir (str | Path | None, optional): Directory containing normal
            RGB images for testing. Defaults to ``None``.
        mask_dir (str | Path | None, optional): Directory containing mask
            annotations. Defaults to ``None``.
        normal_depth_dir (str | Path | None, optional): Directory containing
            normal depth images. Defaults to ``None``.
        abnormal_depth_dir (str | Path | None, optional): Directory containing
            abnormal depth images. Defaults to ``None``.
        normal_test_depth_dir (str | Path | None, optional): Directory containing
            normal depth images for testing. If ``None``, uses split from
            ``normal_dir``. Defaults to ``None``.
        extensions (tuple[str, ...] | None, optional): Image file extensions to
            read. Defaults to ``None``.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Evaluation batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers for data loading.
            Defaults to ``8``.
        train_augmentations (Transform | None): Augmentations to apply dto the training images
            Defaults to ``None``.
        val_augmentations (Transform | None): Augmentations to apply to the validation images.
            Defaults to ``None``.
        test_augmentations (Transform | None): Augmentations to apply to the test images.
            Defaults to ``None``.
        augmentations (Transform | None): General augmentations to apply if stage-specific
            augmentations are not provided.
        test_split_mode (TestSplitMode | str, optional): Method to create test
            set. Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float, optional): Fraction of data for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode | str, optional): Method to create validation
            set. Defaults to ``ValSplitMode.FROM_TEST``.
        val_split_ratio (float, optional): Fraction of data for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Random seed for splitting.
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
        self._name = name
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
        """Set up train and test datasets.

        Args:
            _stage (str | None, optional): Stage of setup. Not used.
                Defaults to ``None``.
        """
        self.train_data = Folder3DDataset(
            name=self.name,
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
        """Get name of the datamodule.

        Returns:
            str: Name of the datamodule.
        """
        return self._name
