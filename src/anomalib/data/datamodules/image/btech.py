"""BTech Data Module.

This module provides a PyTorch Lightning DataModule for the BTech dataset. If the
dataset is not available locally, it will be downloaded and extracted
automatically.

Example:
    Create a BTech datamodule::

        >>> from anomalib.data import BTech
        >>> datamodule = BTech(
        ...     root="./datasets/BTech",
        ...     category="01"
        ... )

Notes:
    The dataset will be automatically downloaded and converted to the required
    format when first used. The directory structure after preparation will be::

        datasets/
        └── BTech/
            ├── 01/
            ├── 02/
            └── 03/

License:
    BTech dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0).
    https://creativecommons.org/licenses/by-nc-sa/4.0/

Reference:
    Mishra, Pankaj, et al. "BTAD—A Large Scale Dataset and Benchmark for
    Real-World Industrial Anomaly Detection." Pattern Recognition 136 (2024):
    109542.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import shutil
from pathlib import Path

import cv2
from torchvision.transforms.v2 import Transform
from tqdm import tqdm

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.btech import BTechDataset
from anomalib.data.utils import DownloadInfo, Split, TestSplitMode, ValSplitMode, download_and_extract

logger = logging.getLogger(__name__)

DOWNLOAD_INFO = DownloadInfo(
    name="btech",
    url="https://avires.dimi.uniud.it/papers/btad/btad.zip",
    hashsum="461c9387e515bfed41ecaae07c50cf6b10def647b36c9e31d239ab2736b10d2a",
)


class BTech(AnomalibDataModule):
    """BTech Lightning Data Module.

    Args:
        root (Path | str): Path to the root of the dataset.
            Defaults to ``"./datasets/BTech"``.
        category (str): Category of the BTech dataset (e.g. ``"01"``, ``"02"``,
            or ``"03"``).
            Defaults to ``"01"``.
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
        To create the BTech datamodule, instantiate the class and call
        ``setup``::

            >>> from anomalib.data import BTech
            >>> datamodule = BTech(
            ...     root="./datasets/BTech",
            ...     category="01",
            ...     train_batch_size=32,
            ...     eval_batch_size=32,
            ...     num_workers=8,
            ... )
            >>> datamodule.setup()

        Get the train dataloader and first batch::

            >>> i, data = next(enumerate(datamodule.train_dataloader()))
            >>> data.keys()
            dict_keys(['image'])
            >>> data["image"].shape
            torch.Size([32, 3, 256, 256])

        Access the validation dataloader and first batch::

            >>> i, data = next(enumerate(datamodule.val_dataloader()))
            >>> data.keys()
            dict_keys(['image_path', 'label', 'mask_path', 'image', 'mask'])
            >>> data["image"].shape, data["mask"].shape
            (torch.Size([32, 3, 256, 256]), torch.Size([32, 256, 256]))

        Access the test dataloader and first batch::

            >>> i, data = next(enumerate(datamodule.test_dataloader()))
            >>> data.keys()
            dict_keys(['image_path', 'label', 'mask_path', 'image', 'mask'])
            >>> data["image"].shape, data["mask"].shape
            (torch.Size([32, 3, 256, 256]), torch.Size([32, 256, 256]))
    """

    def __init__(
        self,
        root: Path | str = "./datasets/BTech",
        category: str = "01",
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
        self.train_data = BTechDataset(
            split=Split.TRAIN,
            root=self.root,
            category=self.category,
        )
        self.test_data = BTechDataset(
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
            Assume the dataset is not available on the file system.
            Here's how the directory structure looks before and after calling
            ``prepare_data``::

                # Before
                $ tree datasets
                datasets
                ├── dataset1
                └── dataset2

                # Calling prepare_data
                >>> datamodule = BTech(root="./datasets/BTech", category="01")
                >>> datamodule.prepare_data()

                # After
                $ tree datasets
                datasets
                ├── dataset1
                ├── dataset2
                └── BTech
                    ├── 01
                    ├── 02
                    └── 03
        """
        if (self.root / self.category).is_dir():
            logger.info("Found the dataset.")
        else:
            download_and_extract(self.root.parent, DOWNLOAD_INFO)

            # rename folder and convert images
            logger.info("Renaming the dataset directory")
            shutil.move(
                src=str(self.root.parent / "BTech_Dataset_transformed"),
                dst=str(self.root),
            )
            logger.info("Convert the bmp formats to png for consistent extensions")
            for filename in tqdm(self.root.glob("**/*.bmp"), desc="Converting"):
                image = cv2.imread(str(filename))
                cv2.imwrite(str(filename.with_suffix(".png")), image)
                filename.unlink()
