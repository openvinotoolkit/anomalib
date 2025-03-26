"""VAD Data Module.

This module provides a PyTorch Lightning DataModule for the VAD dataset. If
the dataset is not available locally, it will be downloaded and extracted
automatically.

Example:
    Create a VAD datamodule::

        >>> from anomalib.data import VAD
        >>> datamodule = VAD(
        ...     root="./datasets/VAD",
        ...     category="vad"
        ... )

Notes:
    The dataset will be automatically downloaded and converted to the required
    format when first used. The directory structure after preparation will be::

        datasets/
        └── VAD/
            └── vad/

License:
    VAD dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0).
    https://creativecommons.org/licenses/by-nc-sa/4.0/

Reference:
    Aimira Baitieva, David Hurych, Victor Besnier, Olivier Bernard:
    Supervised Anomaly Detection for Complex Industrial Images; in:
    The IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024,
    pp. 17754-17762, DOI: 10.1109/CVPR52733.2024.01681.
"""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets import VADDataset
from anomalib.data.utils import DownloadInfo, Split, TestSplitMode, ValSplitMode, download_and_extract

logger = logging.getLogger(__name__)


DOWNLOAD_INFO = DownloadInfo(
    name="VAD",
    url="https://eu.mydrive.ch/shares/88472/0eb328ff388ac0614b383cdfdeada104/download/452926775-1741958563/VAD.zip",
    hashsum="bfffce213d1649c8dd6b8e3afd8d85512736131e9ec7c5edfa0c26e774b5760b",
)


class VAD(AnomalibDataModule):
    """VAD Datamodule.

    Args:
        root (Path | str): Path to the root of the dataset.
            Defaults to ``"./datasets/VAD"``.
        category (str): Category of the VAD dataset. Defaults to ``"vad"``.
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
        test_split_mode (TestSplitMode): Method to create test set.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float): Fraction of data to use for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode): Method to create validation set.
            Defaults to ``ValSplitMode.SAME_AS_TEST``.
        val_split_ratio (float): Fraction of data to use for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Seed for reproducibility.
            Defaults to ``None``.

    Example:
        Create VAD datamodule with default settings::

            >>> datamodule = VAD()
            >>> datamodule.setup()
            >>> i, data = next(enumerate(datamodule.train_dataloader()))
            >>> data.keys()
            dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])

            >>> data["image"].shape
            torch.Size([32, 3, 256, 256])

        Create validation set from test data::

            >>> datamodule = VAD(
            ...     val_split_mode=ValSplitMode.FROM_TEST,
            ...     val_split_ratio=0.1
            ... )

        Create synthetic validation set::

            >>> datamodule = VAD(
            ...     val_split_mode=ValSplitMode.SYNTHETIC,
            ...     val_split_ratio=0.2
            ... )
    """

    def __init__(
        self,
        root: Path | str = "./datasets/VAD",
        category: str = "vad",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode | str = ValSplitMode.SAME_AS_TEST,
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

        self.root = Path(root)
        self.category = category

    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets and perform dynamic subset splitting.

        This method may be overridden in subclass for custom splitting behaviour.

        Note:
            The stage argument is not used here. This is because, for a given
            instance of an AnomalibDataModule subclass, all three subsets are
            created at the first call of setup(). This is to accommodate the
            subset splitting behaviour of anomaly tasks, where the validation set
            is usually extracted from the test set, and the test set must
            therefore be created as early as the `fit` stage.
        """
        self.train_data = VADDataset(
            split=Split.TRAIN,
            root=self.root,
            category=self.category,
        )
        self.test_data = VADDataset(
            split=Split.TEST,
            root=self.root,
            category=self.category,
        )

    def prepare_data(self) -> None:
        """Download the dataset if not available.

        This method checks if the specified dataset is available in the file
        system. If not, it downloads and extracts the dataset into the
        appropriate directory.

        Example:
            Assume the dataset is not available on the file system::

                >>> datamodule = VAD(
                ...     root="./datasets/VAD",
                ...     category="vad"
                ... )
                >>> datamodule.prepare_data()

            Directory structure after download::

                datasets/
                    └── VAD/
                        └── vad/
        """
        if (self.root / self.category).is_dir():
            logger.info("Found the dataset.")
        else:
            download_and_extract(self.root, DOWNLOAD_INFO)
