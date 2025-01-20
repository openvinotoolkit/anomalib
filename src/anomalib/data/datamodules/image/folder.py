"""Custom Folder Data Module.

This script creates a custom Lightning DataModule from a folder containing normal
and abnormal images.

Example:
    Create a folder datamodule::

        >>> from anomalib.data import Folder
        >>> datamodule = Folder(
        ...     name="custom_folder",
        ...     root="./datasets/custom",
        ...     normal_dir="good",
        ...     abnormal_dir="defect"
        ... )

Notes:
    The directory structure should be organized as follows::

        root/
        ├── normal_dir/
        │   ├── image1.png
        │   └── image2.png
        ├── abnormal_dir/
        │   ├── image3.png
        │   └── image4.png
        └── mask_dir/
            ├── mask3.png
            └── mask4.png
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from pathlib import Path

from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.folder import FolderDataset
from anomalib.data.utils import Split, TestSplitMode, ValSplitMode


class Folder(AnomalibDataModule):
    """Folder DataModule.

    Args:
        name (str): Name of the dataset. Used for logging/saving.
        normal_dir (str | Path | Sequence): Directory containing normal images.
        root (str | Path | None): Root folder containing normal and abnormal
            directories. Defaults to ``None``.
        abnormal_dir (str | Path | None | Sequence): Directory containing
            abnormal images. Defaults to ``None``.
        normal_test_dir (str | Path | Sequence | None): Directory containing
            normal test images. Defaults to ``None``.
        mask_dir (str | Path | Sequence | None): Directory containing mask
            annotations. Defaults to ``None``.
        normal_split_ratio (float): Ratio to split normal training images for
            test set when no normal test images exist.
            Defaults to ``0.2``.
        extensions (tuple[str, ...] | None): Image extensions to include.
            Defaults to ``None``.
        train_batch_size (int): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int): Validation/test batch size.
            Defaults to ``32``.
        num_workers (int): Number of workers for data loading.
            Defaults to ``8``.
        train_augmentations (Transform | None): Augmentations to apply dto the training images
            Defaults to ``None``.
        val_augmentations (Transform | None): Augmentations to apply to the validation images.
            Defaults to ``None``.
        test_augmentations (Transform | None): Augmentations to apply to the test images.
            Defaults to ``None``.
        augmentations (Transform | None): General augmentations to apply if stage-specific
            augmentations are not provided.
        test_split_mode (TestSplitMode): Method to obtain test subset.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float): Fraction of train images for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode): Method to obtain validation subset.
            Defaults to ``ValSplitMode.FROM_TEST``.
        val_split_ratio (float): Fraction of images for validation.
            Defaults to ``0.5``.
        seed (int | None): Random seed for splitting.
            Defaults to ``None``.

    Example:
        Create and setup a folder datamodule::

            >>> from anomalib.data import Folder
            >>> datamodule = Folder(
            ...     name="custom",
            ...     root="./datasets/custom",
            ...     normal_dir="good",
            ...     abnormal_dir="defect",
            ...     mask_dir="mask"
            ... )
            >>> datamodule.setup()

        Get a batch from train dataloader::

            >>> batch = next(iter(datamodule.train_dataloader()))
            >>> batch.keys()
            dict_keys(['image', 'label', 'mask', 'image_path', 'mask_path'])

        Get a batch from test dataloader::

            >>> batch = next(iter(datamodule.test_dataloader()))
            >>> batch.keys()
            dict_keys(['image', 'label', 'mask', 'image_path', 'mask_path'])
    """

    def __init__(
        self,
        name: str,
        normal_dir: str | Path | Sequence[str | Path],
        root: str | Path | None = None,
        abnormal_dir: str | Path | Sequence[str | Path] | None = None,
        normal_test_dir: str | Path | Sequence[str | Path] | None = None,
        mask_dir: str | Path | Sequence[str | Path] | None = None,
        normal_split_ratio: float = 0.2,
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
        self._name = name
        self.root = root
        self.normal_dir = normal_dir
        self.abnormal_dir = abnormal_dir
        self.normal_test_dir = normal_test_dir
        self.mask_dir = mask_dir
        self.extensions = extensions
        test_split_mode = TestSplitMode(test_split_mode)
        val_split_mode = ValSplitMode(val_split_mode)
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

        self.normal_split_ratio = normal_split_ratio

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = FolderDataset(
            name=self.name,
            split=Split.TRAIN,
            root=self.root,
            normal_dir=self.normal_dir,
            abnormal_dir=self.abnormal_dir,
            normal_test_dir=self.normal_test_dir,
            mask_dir=self.mask_dir,
            extensions=self.extensions,
        )

        self.test_data = FolderDataset(
            name=self.name,
            split=Split.TEST,
            root=self.root,
            normal_dir=self.normal_dir,
            abnormal_dir=self.abnormal_dir,
            normal_test_dir=self.normal_test_dir,
            mask_dir=self.mask_dir,
            extensions=self.extensions,
        )

    @property
    def name(self) -> str:
        """Get name of the datamodule.

        Returns:
            Name of the datamodule.
        """
        return self._name
