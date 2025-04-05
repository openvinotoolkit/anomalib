"""MVTec AD 2 Lightning Data Module.

This module implements a PyTorch Lightning DataModule for the MVTec AD 2 dataset.
The module handles downloading, loading, and preprocessing of the dataset for
training and evaluation.

The dataset provides three different test sets:
    - Public test set (test_public/): Contains both normal and anomalous samples with ground truth masks
      for facilitating local testing and initial performance estimation
    - Private test set (test_private/): Official unseen test set without ground truth
      for entering the leaderboard
    - Private mixed test set (test_private_mixed/): Contains unseen test samples captured
      under seen and unseen lighting conditions (mixed randomly) without ground truth

The public test set is meant for local evaluation, while the private test sets
are the official test sets for entering the leaderboard on the evaluation server
(https://benchmark.mvtec.com/).

License:
    MVTec AD 2 dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0) https://creativecommons.org/licenses/by-nc-sa/4.0/

Reference:
    Lars Heckler-Kram, Jan-Hendrik Neudeck, Ulla Scheler, Rebecca König, Carsten Steger:
    The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection.
    arXiv preprint, 2024 (to appear).
"""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image import MVTecAD2Dataset
from anomalib.data.datasets.image.mvtecad2 import TestType
from anomalib.data.utils import DownloadInfo, Split, download_and_extract

logger = logging.getLogger(__name__)

# Download information for MVTec AD 2 dataset
DOWNLOAD_INFO = DownloadInfo(
    name="mvtecad2",
    url="https://www.mydrive.ch/shares/121573/7f68fe2c4f7c2ceaa08f463aaeb2f414/download/"
    "466712769-1743422799/mvtec_ad_2.tar.gz",
    hashsum="c0ded99ef32bfc8e352d52beb44515e5b292b8598cb963aadfa91ca0763505e4",
)


class MVTecAD2(AnomalibDataModule):
    """MVTec AD 2 Lightning Data Module.

    Args:
        root (str | Path): Path to the dataset root directory.
            Defaults to ``"./datasets/MVTec_AD_2"``.
        category (str): Name of the MVTec AD 2 category to load.
            Defaults to ``"sheet_metal"``.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Validation and test batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers for data loading.
            Defaults to ``8``.
        train_augmentations (Transform | None): Augmentations to apply to the training images
            Defaults to ``None``.
        val_augmentations (Transform | None): Augmentations to apply to the validation images.
            Defaults to ``None``.
        test_augmentations (Transform | None): Augmentations to apply to the test images.
            Defaults to ``None``.
        augmentations (Transform | None): General augmentations to apply if stage-specific
            augmentations are not provided.
        test_type (str | TestType): Type of test set to use:
            - ``"public"``: Test set with ground truth for local evaluation and initial
              performance estimation
            - ``"private"``: Official test set without ground truth for leaderboard submission
            - ``"private_mixed"``: Official test set with mixed lighting conditions (seen and
              unseen, randomly mixed) for leaderboard submission
            Defaults to ``TestType.PUBLIC``.
        seed (int | None, optional): Random seed for reproducibility.
            Defaults to ``None``.

    Example:
        >>> from anomalib.data import MVTecAD2
        >>> datamodule = MVTecAD2(
        ...     root="./datasets/MVTec_AD_2",
        ...     category="sheet_metal",
        ...     train_batch_size=32,
        ...     eval_batch_size=32,
        ...     num_workers=8,
        ... )

        To use private test set:
        >>> datamodule = MVTecAD2(
        ...     root="./datasets/MVTec_AD_2",
        ...     category="sheet_metal",
        ...     test_type="private",
        ... )

        Access different test sets:
        >>> datamodule.setup()
        >>> public_loader = datamodule.test_dataloader()  # returns loader based on test_type
        >>> private_loader = datamodule.test_dataloader(test_type="private")
        >>> mixed_loader = datamodule.test_dataloader(test_type="private_mixed")
    """

    def __init__(
        self,
        root: str | Path = "./datasets/MVTec_AD_2",
        category: str = "sheet_metal",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        test_type: str | TestType = TestType.PUBLIC,
        seed: int | None = None,
    ) -> None:
        """Initialize MVTec AD 2 datamodule."""
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            train_augmentations=train_augmentations,
            val_augmentations=val_augmentations,
            test_augmentations=test_augmentations,
            augmentations=augmentations,
            seed=seed,
        )

        self.root = Path(root)
        self.category = category
        self.test_type = TestType(test_type) if isinstance(test_type, str) else test_type

    def prepare_data(self) -> None:
        """Download the dataset if not available.

        This method checks if the specified dataset is available in the file
        system. If not, it downloads and extracts the dataset into the
        appropriate directory.

        Example:
            Assume the dataset is not available on the file system::

                >>> datamodule = MVTecAD2(
                ...     root="./datasets/MVTecAD2",
                ...     category="can"
                ... )
                >>> datamodule.prepare_data()

            Directory structure after download::

                datasets/
                └── MVTecAD2/
                    ├── can/
                    ├── fabric/
                    └── ...
        """
        if (self.root / self.category).is_dir():
            logger.info("Found the dataset.")
        else:
            download_and_extract(self.root, DOWNLOAD_INFO)

    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets and perform train/validation/test split.

        Args:
            _stage: str | None: Optional argument for compatibility with pytorch
                lightning. Defaults to None.
        """
        self.train_data = MVTecAD2Dataset(
            root=self.root,
            category=self.category,
            split=Split.TRAIN,
            augmentations=self.train_augmentations,
        )

        # MVTec AD 2 has a dedicated validation set
        self.val_data = MVTecAD2Dataset(
            root=self.root,
            category=self.category,
            split=Split.VAL,
            augmentations=self.val_augmentations,
        )

        # Create datasets for all test types
        self.test_public_data = MVTecAD2Dataset(
            root=self.root,
            category=self.category,
            split=Split.TEST,
            test_type=TestType.PUBLIC,
            augmentations=self.test_augmentations,
        )

        self.test_private_data = MVTecAD2Dataset(
            root=self.root,
            category=self.category,
            split=Split.TEST,
            test_type=TestType.PRIVATE,
            augmentations=self.test_augmentations,
        )

        self.test_private_mixed_data = MVTecAD2Dataset(
            root=self.root,
            category=self.category,
            split=Split.TEST,
            test_type=TestType.PRIVATE_MIXED,
            augmentations=self.test_augmentations,
        )

        # Always set test_data to public test set for standard evaluation
        self.test_data = self.test_public_data

    def test_dataloader(self, test_type: str | TestType | None = None) -> EVAL_DATALOADERS:
        """Get test dataloader for the specified test type.

        Args:
            test_type (str | TestType | None, optional): Type of test set to use:
                - ``"public"``: Test set with ground truth for local evaluation
                - ``"private"``: Official test set without ground truth for leaderboard
                - ``"private_mixed"``: Official test set with mixed lighting conditions
                If None, uses the test_type specified in __init__.
                Defaults to None.

        Example:
            >>> datamodule.setup()
            >>> public_loader = datamodule.test_dataloader()  # returns loader based on test_type
            >>> private_loader = datamodule.test_dataloader(test_type="private")
            >>> mixed_loader = datamodule.test_dataloader(test_type="private_mixed")

        Returns:
            EVAL_DATALOADERS: Test dataloader for the specified test type.
        """
        test_type = test_type or self.test_type
        test_type = TestType(test_type) if isinstance(test_type, str) else test_type

        if test_type == TestType.PUBLIC:
            dataset = self.test_public_data
        elif test_type == TestType.PRIVATE:
            dataset = self.test_private_data
        elif test_type == TestType.PRIVATE_MIXED:
            dataset = self.test_private_mixed_data
        else:
            msg = f"Invalid test type: {test_type}. Must be one of {TestType.__members__.keys()}."
            raise ValueError(msg)

        return DataLoader(
            dataset=dataset,
            shuffle=False,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_fn,
        )
