"""DataModule for Datumaro format.

This module provides a PyTorch Lightning DataModule for datasets in Datumaro
format. Currently only supports annotations exported from Intel Geti™.

Example:
    Create a Datumaro datamodule::

        >>> from pathlib import Path
        >>> from anomalib.data import Datumaro
        >>> datamodule = Datumaro(
        ...     root="./datasets/datumaro",
        ...     train_batch_size=32,
        ...     eval_batch_size=32,
        ...     num_workers=8,
        ... )
        >>> datamodule.setup()
        >>> i, data = next(enumerate(datamodule.train_dataloader()))
        >>> data.keys()
        dict_keys(['image_path', 'label', 'image'])

Notes:
    The directory structure should be organized as follows::

        root/
        ├── annotations/
        │   ├── train.json
        │   └── test.json
        └── images/
            ├── train/
            │   ├── image1.jpg
            │   └── image2.jpg
            └── test/
                ├── image3.jpg
                └── image4.jpg
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base import AnomalibDataModule
from anomalib.data.datasets.image.datumaro import DatumaroDataset
from anomalib.data.utils import Split, TestSplitMode, ValSplitMode


class Datumaro(AnomalibDataModule):
    """Datumaro datamodule.

    Args:
        root (Path | str): Path to the dataset root directory.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Test batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            Defaults to ``8``.
        train_augmentations (Transform | None): Augmentations to apply dto the training images
            Defaults to ``None``.
        val_augmentations (Transform | None): Augmentations to apply to the validation images.
            Defaults to ``None``.
        test_augmentations (Transform | None): Augmentations to apply to the test images.
            Defaults to ``None``.
        augmentations (Transform | None): General augmentations to apply if stage-specific
            augmentations are not provided.
        image_size (tuple[int, int], optional): Size to which input images should be resized.
            Defaults to ``None``.
        test_split_mode (TestSplitMode): Setting that determines how the testing
            subset is obtained.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float): Fraction of images from the train set that will
            be reserved for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode): Setting that determines how the validation
            subset is obtained.
            Defaults to ``ValSplitMode.SAME_AS_TEST``.
        val_split_ratio (float): Fraction of train or test images that will be
            reserved for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Seed which may be set to a fixed value for
            reproducibility.
            Defaults to ``None``.

    Example:
        >>> from anomalib.data import Datumaro
        >>> datamodule = Datumaro(
        ...     root="./datasets/datumaro",
        ...     train_batch_size=32,
        ...     eval_batch_size=32,
        ...     num_workers=8,
        ... )
        >>> datamodule.setup()
        >>> i, data = next(enumerate(datamodule.train_dataloader()))
        >>> data.keys()
        dict_keys(['image_path', 'label', 'image'])
    """

    def __init__(
        self,
        root: str | Path,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.5,
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
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            seed=seed,
        )
        self.root = root

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = DatumaroDataset(
            root=self.root,
            split=Split.TRAIN,
        )
        self.test_data = DatumaroDataset(
            root=self.root,
            split=Split.TEST,
        )
