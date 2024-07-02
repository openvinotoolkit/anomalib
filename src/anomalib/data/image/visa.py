"""Visual Anomaly (VisA) Dataset (CC BY-NC-SA 4.0).

Description:
    This script contains PyTorch Dataset, Dataloader and PyTorch
        Lightning DataModule for the Visual Anomal (VisA) dataset.
    If the dataset is not on the file system, the script downloads and
        extracts the dataset and create PyTorch data objects.
License:
    The VisA dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0)(https://creativecommons.org/licenses/by-nc-sa/4.0/).
Reference:
    - Zou, Y., Jeong, J., Pemula, L., Zhang, D., & Dabeer, O. (2022). SPot-the-Difference
      Self-supervised Pre-training for Anomaly Detection and Segmentation. In European
      Conference on Computer Vision (pp. 392-408). Springer, Cham.
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Subset splitting code adapted from https://github.com/amazon-science/spot-diff
# Original licence: Apache-2.0

import csv
import logging
import shutil
from pathlib import Path

import cv2
from torchvision.transforms.v2 import Transform

from anomalib import TaskType
from anomalib.data.base import AnomalibDataModule, AnomalibDataset
from anomalib.data.utils import (
    DownloadInfo,
    Split,
    TestSplitMode,
    ValSplitMode,
    download_and_extract,
)

from .mvtec import make_mvtec_dataset

logger = logging.getLogger(__name__)

EXTENSIONS = (".png", ".jpg", ".JPG")

DOWNLOAD_INFO = DownloadInfo(
    name="VisA",
    url="https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar",
    hashsum="2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362",
)

CATEGORIES = (
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
)


class VisaDataset(AnomalibDataset):
    """VisA dataset class.

    Args:
        task (TaskType): Task type, ``classification``, ``detection`` or ``segmentation``
        root (str | Path): Path to the root of the dataset
        category (str): Sub-category of the dataset, e.g. 'candle'
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
        split (str | Split | None): Split of the dataset, usually Split.TRAIN or Split.TEST
            Defaults to ``None``.

    Examples:
        To create a Visa dataset for classification:

        .. code-block:: python

            from anomalib.data.image.visa import VisaDataset
            from anomalib.data.utils.transforms import get_transforms

            transform = get_transforms(image_size=256)
            dataset = VisaDataset(
                task="classification",
                transform=transform,
                split="train",
                root="./datasets/visa/visa_pytorch/",
                category="candle",
            )
            dataset.setup()
            dataset[0].keys()

            # Output
            dict_keys(['image_path', 'label', 'image'])

        If you want to use the dataset for segmentation, you can use the same
        code as above, with the task set to ``segmentation``. The dataset will
        then have a ``mask`` key in the output dictionary.

        .. code-block:: python

            from anomalib.data.image.visa import VisaDataset
            from anomalib.data.utils.transforms import get_transforms

            transform = get_transforms(image_size=256)
            dataset = VisaDataset(
                task="segmentation",
                transform=transform,
                split="train",
                root="./datasets/visa/visa_pytorch/",
                category="candle",
            )
            dataset.setup()
            dataset[0].keys()

            # Output
            dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])

    """

    def __init__(
        self,
        task: TaskType,
        root: str | Path,
        category: str,
        transform: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(task=task, transform=transform)

        self.root_category = Path(root) / category
        self.split = split
        self.samples = make_mvtec_dataset(self.root_category, split=self.split, extensions=EXTENSIONS)


class Visa(AnomalibDataModule):
    """VisA Datamodule.

    Args:
        root (Path | str): Path to the root of the dataset
            Defaults to ``"./datasets/visa"``.
        category (str): Category of the Visa dataset such as ``candle``.
            Defaults to ``"candle"``.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Test batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            Defaults to ``8``.
        task (TaskType): Task type, 'classification', 'detection' or 'segmentation'
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
            Defaults to ``ValSplitMode.SAME_AS_TEST``.
        val_split_ratio (float): Fraction of train or test images that will be reserved for validation.
            Defatuls to ``0.5``.
        seed (int | None, optional): Seed which may be set to a fixed value for reproducibility.
            Defaults to ``None``.
    """

    def __init__(
        self,
        root: Path | str = "./datasets/visa",
        category: str = "capsules",
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
        val_split_mode: ValSplitMode | str = ValSplitMode.SAME_AS_TEST,
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

        self.task = TaskType(task)
        self.root = Path(root)
        self.split_root = self.root / "visa_pytorch"
        self.category = category

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = VisaDataset(
            task=self.task,
            transform=self.train_transform,
            split=Split.TRAIN,
            root=self.split_root,
            category=self.category,
        )
        self.test_data = VisaDataset(
            task=self.task,
            transform=self.eval_transform,
            split=Split.TEST,
            root=self.split_root,
            category=self.category,
        )

    def prepare_data(self) -> None:
        """Download the dataset if not available.

        This method checks if the specified dataset is available in the file system.
        If not, it downloads and extracts the dataset into the appropriate directory.

        Example:
            Assume the dataset is not available on the file system.
            Here's how the directory structure looks before and after calling the
            `prepare_data` method:

            Before:

            .. code-block:: bash

                $ tree datasets
                datasets
                ├── dataset1
                └── dataset2

            Calling the method:

            .. code-block:: python

                >> datamodule = Visa()
                >> datamodule.prepare_data()

            After:

            .. code-block:: bash

                $ tree datasets
                datasets
                ├── dataset1
                ├── dataset2
                └── visa
                    ├── candle
                    ├── ...
                    ├── pipe_fryum
                    │   ├── Data
                    │   └── image_anno.csv
                    ├── split_csv
                    │   ├── 1cls.csv
                    │   ├── 2cls_fewshot.csv
                    │   └── 2cls_highshot.csv
                    ├── VisA_20220922.tar
                    └── visa_pytorch
                        ├── candle
                        ├── ...
                        ├── pcb4
                        └── pipe_fryum

            ``prepare_data`` ensures that the dataset is converted to MVTec
            format. ``visa_pytorch`` is the directory that contains the dataset
            in the MVTec format. ``visa`` is the directory that contains the
            original dataset.
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

        adapted from https://github.com/amazon-science/spot-diff
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
