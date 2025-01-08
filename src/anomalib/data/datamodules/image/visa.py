"""Visual Anomaly (VisA) Data Module.

This module provides a PyTorch Lightning DataModule for the Visual Anomaly (VisA)
dataset. If the dataset is not available locally, it will be downloaded and
extracted automatically.

Example:
    Create a VisA datamodule::

        >>> from anomalib.data import Visa
        >>> datamodule = Visa(
        ...     root="./datasets/visa",
        ...     category="capsules"
        ... )

Notes:
    The dataset will be automatically downloaded and converted to the required
    format when first used. The directory structure after preparation will be::

        datasets/
        └── visa/
            ├── visa_pytorch/
            │   ├── candle/
            │   ├── capsules/
            │   └── ...
            └── VisA_20220922.tar

License:
    The VisA dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0).
    https://creativecommons.org/licenses/by-nc-sa/4.0/

Reference:
    Zou, Y., Jeong, J., Pemula, L., Zhang, D., & Dabeer, O. (2022).
    SPot-the-Difference Self-supervised Pre-training for Anomaly Detection
    and Segmentation. In European Conference on Computer Vision (pp. 392-408).
    Springer, Cham.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Subset splitting code adapted from https://github.com/amazon-science/spot-diff
# Original licence: Apache-2.0

import csv
import logging
import shutil
from pathlib import Path

import cv2
from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.visa import VisaDataset
from anomalib.data.utils import DownloadInfo, Split, TestSplitMode, ValSplitMode, download_and_extract

logger = logging.getLogger(__name__)

DOWNLOAD_INFO = DownloadInfo(
    name="VisA",
    url="https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar",
    hashsum="2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362",
)


class Visa(AnomalibDataModule):
    """VisA Datamodule.

    Args:
        root (Path | str): Path to the root of the dataset.
            Defaults to ``"./datasets/visa"``.
        category (str): Category of the VisA dataset (e.g. ``"candle"``).
            Defaults to ``"capsules"``.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Test batch size.
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
        test_split_mode (TestSplitMode | str): Method to create test set.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float): Fraction of data to use for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode | str): Method to create validation set.
            Defaults to ``ValSplitMode.SAME_AS_TEST``.
        val_split_ratio (float): Fraction of data to use for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Random seed for reproducibility.
            Defaults to ``None``.
    """

    def __init__(
        self,
        root: Path | str = "./datasets/visa",
        category: str = "capsules",
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
        self.split_root = self.root / "visa_pytorch"
        self.category = category

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = VisaDataset(
            split=Split.TRAIN,
            root=self.split_root,
            category=self.category,
        )
        self.test_data = VisaDataset(
            split=Split.TEST,
            root=self.split_root,
            category=self.category,
        )

    def prepare_data(self) -> None:
        """Download and prepare the dataset if not available.

        This method checks if the dataset exists and is properly formatted.
        If not, it downloads and prepares the data in the following steps:

        1. If the processed dataset exists (``visa_pytorch/{category}``), do
           nothing
        2. If the raw dataset exists but isn't processed, apply the train/test
           split
        3. If the dataset doesn't exist, download, extract, and process it

        The final directory structure will be::

            datasets/
            └── visa/
                ├── visa_pytorch/
                │   ├── candle/
                │   │   ├── train/
                │   │   │   └── good/
                │   │   ├── test/
                │   │   │   ├── good/
                │   │   │   └── bad/
                │   │   └── ground_truth/
                │   │       └── bad/
                │   └── ...
                └── VisA_20220922.tar
        """
        if (self.split_root / self.category).is_dir():
            # dataset is available, and split has been applied
            logger.info("Found the dataset and train/test split.")
        elif (self.root / self.category).is_dir():
            # dataset is available, but split has not yet been applied
            logger.info("Found the dataset. Applying train/test split.")
            self.apply_cls1_split()
        else:
            # dataset is not available
            download_and_extract(self.root, DOWNLOAD_INFO)
            logger.info("Downloaded the dataset. Applying train/test split.")
            self.apply_cls1_split()

    def apply_cls1_split(self) -> None:
        """Apply the 1-class subset splitting using the fixed split in the csv file.

        Adapted from https://github.com/amazon-science/spot-diff.
        """
        logger.info("preparing data")
        categories = [
            "candle",
            "capsules",
            "cashew",
            "chewinggum",
            "fryum",
            "macaroni1",
            "macaroni2",
            "pcb1",
            "pcb2",
            "pcb3",
            "pcb4",
            "pipe_fryum",
        ]

        split_file = self.root / "split_csv" / "1cls.csv"

        for category in categories:
            train_folder = self.split_root / category / "train"
            test_folder = self.split_root / category / "test"
            mask_folder = self.split_root / category / "ground_truth"

            train_img_good_folder = train_folder / "good"
            test_img_good_folder = test_folder / "good"
            test_img_bad_folder = test_folder / "bad"
            test_mask_bad_folder = mask_folder / "bad"

            train_img_good_folder.mkdir(parents=True, exist_ok=True)
            test_img_good_folder.mkdir(parents=True, exist_ok=True)
            test_img_bad_folder.mkdir(parents=True, exist_ok=True)
            test_mask_bad_folder.mkdir(parents=True, exist_ok=True)

        with split_file.open(encoding="utf-8") as file:
            csvreader = csv.reader(file)
            next(csvreader)
            for row in csvreader:
                category, split, label, image_path, mask_path = row
                label = "good" if label == "normal" else "bad"
                image_name = image_path.split("/")[-1]
                mask_name = mask_path.split("/")[-1]

                img_src_path = self.root / image_path
                msk_src_path = self.root / mask_path
                img_dst_path = self.split_root / category / split / label / image_name
                msk_dst_path = self.split_root / category / "ground_truth" / label / mask_name

                shutil.copyfile(img_src_path, img_dst_path)
                if split == "test" and label == "bad":
                    mask = cv2.imread(str(msk_src_path))

                    # binarize mask
                    mask[mask != 0] = 255

                    cv2.imwrite(str(msk_dst_path), mask)
