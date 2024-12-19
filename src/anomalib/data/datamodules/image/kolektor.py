"""Kolektor Surface-Defect Data Module.

Description:
    This script provides a PyTorch DataModule for the Kolektor Surface-Defect
    dataset. The dataset can be accessed at `Kolektor Surface-Defect Dataset
    <https://www.vicos.si/resources/kolektorsdd/>`_.

License:
    The Kolektor Surface-Defect dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0). For more details, visit `Creative Commons License
    <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_.

Reference:
    Tabernik, Domen, Samo Šela, Jure Skvarč, and Danijel Skočaj.
    "Segmentation-based deep-learning approach for surface-defect detection."
    Journal of Intelligent Manufacturing 31, no. 3 (2020): 759-776.
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.kolektor import KolektorDataset
from anomalib.data.utils import DownloadInfo, Split, TestSplitMode, ValSplitMode, download_and_extract

logger = logging.getLogger(__name__)

DOWNLOAD_INFO = DownloadInfo(
    name="kolektor",
    url="https://go.vicos.si/kolektorsdd",
    hashsum="65dc621693418585de9c4467d1340ea7958a6181816f0dc2883a1e8b61f9d4dc",
    filename="KolektorSDD.zip",
)


class Kolektor(AnomalibDataModule):
    """Kolektor Surface-Defect DataModule.

    Args:
        root (Path | str): Path to the root of the dataset.
            Defaults to ``"./datasets/kolektor"``.
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

    Example:
        >>> from anomalib.data import Kolektor
        >>> datamodule = Kolektor(
        ...     root="./datasets/kolektor",
        ...     train_batch_size=32,
        ...     eval_batch_size=32,
        ...     num_workers=8,
        ... )
        >>> datamodule.setup()
        >>> i, data = next(enumerate(datamodule.train_dataloader()))
        >>> data.keys()
        dict_keys(['image', 'label', 'mask', 'image_path', 'mask_path'])
    """

    def __init__(
        self,
        root: Path | str = "./datasets/kolektor",
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

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = KolektorDataset(
            split=Split.TRAIN,
            root=self.root,
        )
        self.test_data = KolektorDataset(
            split=Split.TEST,
            root=self.root,
        )

    def prepare_data(self) -> None:
        """Download the dataset if not available.

        This method checks if the specified dataset is available in the file
        system. If not, it downloads and extracts the dataset into the
        appropriate directory.

        Example:
            Assume the dataset is not available on the file system.
            Here's how the directory structure looks before and after calling
            the ``prepare_data`` method:

            Before::

                $ tree datasets
                datasets
                ├── dataset1
                └── dataset2

            Calling the method:

            >>> datamodule = Kolektor(root="./datasets/kolektor")
            >>> datamodule.prepare_data()

            After::

                $ tree datasets
                datasets
                ├── dataset1
                ├── dataset2
                └── kolektor
                    ├── kolektorsdd
                    ├── kos01
                    ├── ...
                    └── kos50
                        ├── Part0.jpg
                        ├── Part0_label.bmp
                        └── ...
        """
        if (self.root).is_dir():
            logger.info("Found the dataset.")
        else:
            download_and_extract(self.root, DOWNLOAD_INFO)
