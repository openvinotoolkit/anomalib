"""Kolektor Surface-Defect Dataset (CC BY-NC-SA 4.0).

Description:
    This script provides a PyTorch Dataset, DataLoader, and PyTorch Lightning DataModule for the Kolektor
    Surface-Defect dataset. The dataset can be accessed at `Kolektor Surface-Defect Dataset <https://www.vicos.si/resources/kolektorsdd/>`_.

License:
    The Kolektor Surface-Defect dataset is released under the Creative Commons Attribution-NonCommercial-ShareAlike
    4.0 International License (CC BY-NC-SA 4.0). For more details, visit
    `Creative Commons License <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_.

Reference:
    Tabernik, Domen, Samo Šela, Jure Skvarč, and Danijel Skočaj. "Segmentation-based deep-learning approach
    for surface-defect detection." Journal of Intelligent Manufacturing 31, no. 3 (2020): 759-776.
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

import numpy as np
from cv2 import imread
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torchvision.transforms.v2 import Transform

from anomalib import TaskType
from anomalib.data.base import AnomalibDataModule, AnomalibDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import (
    DownloadInfo,
    Split,
    TestSplitMode,
    ValSplitMode,
    download_and_extract,
    validate_path,
)

__all__ = ["Kolektor", "KolektorDataset", "make_kolektor_dataset"]

logger = logging.getLogger(__name__)

DOWNLOAD_INFO = DownloadInfo(
    name="kolektor",
    url="https://go.vicos.si/kolektorsdd",
    hashsum="65dc621693418585de9c4467d1340ea7958a6181816f0dc2883a1e8b61f9d4dc",
    filename="KolektorSDD.zip",
)


def is_mask_anomalous(path: str) -> int:
    """Check if a mask shows defects.

    Args:
        path (str): Path to the mask file.

    Returns:
        int: 1 if the mask shows defects, 0 otherwise.

    Example:
        Assume that the following image is a mask for a defective image.
        Then the function will return 1.

        >>> from anomalib.data.image.kolektor import is_mask_anomalous
        >>> path = './KolektorSDD/kos01/Part0_label.bmp'
        >>> is_mask_anomalous(path)
        1
    """
    img_arr = imread(path)
    if np.all(img_arr == 0):
        return 0
    return 1


def make_kolektor_dataset(
    root: str | Path,
    train_split_ratio: float = 0.8,
    split: str | Split | None = None,
) -> DataFrame:
    """Create Kolektor samples by parsing the Kolektor data file structure.

    The files are expected to follow this structure:
    - Image files: `path/to/dataset/item/image_filename.jpg`, `path/to/dataset/kos01/Part0.jpg`
    - Mask files: `path/to/dataset/item/mask_filename.bmp`, `path/to/dataset/kos01/Part0_label.bmp`

    This function creates a DataFrame to store the parsed information in the following format:

    +---+-------------------+--------+-------+---------+-----------------------+------------------------+-------------+
    |   | path              | item   | split | label   | image_path            | mask_path              | label_index |
    +---+-------------------+--------+-------+---------+-----------------------+------------------------+-------------+
    | 0 | KolektorSDD       | kos01  | test  | Bad     | /path/to/image_file   | /path/to/mask_file     | 1           |
    +---+-------------------+--------+-------+---------+-----------------------+------------------------+-------------+

    Args:
        root (Path): Path to the dataset.
        train_split_ratio (float, optional): Ratio for splitting good images into train/test sets.
            Defaults to ``0.8``.
        split (str | Split | None, optional): Dataset split (either 'train' or 'test').
            Defaults to ``None``.

    Returns:
        pandas.DataFrame: An output DataFrame containing the samples of the dataset.

    Example:
        The following example shows how to get training samples from the Kolektor Dataset:

        >>> from pathlib import Path
        >>> root = Path('./KolektorSDD/')
        >>> samples = create_kolektor_samples(root, train_split_ratio=0.8)
        >>> samples.head()
               path       item  split label   image_path                    mask_path                   label_index
           0   KolektorSDD   kos01  train Good  KolektorSDD/kos01/Part0.jpg  KolektorSDD/kos01/Part0_label.bmp  0
           1   KolektorSDD   kos01  train Good  KolektorSDD/kos01/Part1.jpg  KolektorSDD/kos01/Part1_label.bmp  0
           2   KolektorSDD   kos01  train Good  KolektorSDD/kos01/Part2.jpg  KolektorSDD/kos01/Part2_label.bmp  0
           3   KolektorSDD   kos01  test  Good  KolektorSDD/kos01/Part3.jpg  KolektorSDD/kos01/Part3_label.bmp  0
           4   KolektorSDD   kos01  train Good  KolektorSDD/kos01/Part4.jpg  KolektorSDD/kos01/Part4_label.bmp  0
    """
    root = validate_path(root)

    # Get list of images and masks
    samples_list = [(str(root),) + f.parts[-2:] for f in root.glob(r"**/*") if f.suffix == ".jpg"]
    masks_list = [(str(root),) + f.parts[-2:] for f in root.glob(r"**/*") if f.suffix == ".bmp"]

    if not samples_list:
        msg = f"Found 0 images in {root}"
        raise RuntimeError(msg)

    # Create dataframes
    samples = DataFrame(samples_list, columns=["path", "item", "image_path"])
    masks = DataFrame(masks_list, columns=["path", "item", "image_path"])

    # Modify image_path column by converting to absolute path
    samples["image_path"] = samples.path + "/" + samples.item + "/" + samples.image_path
    masks["image_path"] = masks.path + "/" + masks.item + "/" + masks.image_path

    # Sort samples by image path
    samples = samples.sort_values(by="image_path", ignore_index=True)
    masks = masks.sort_values(by="image_path", ignore_index=True)

    # Add mask paths for sample images
    samples["mask_path"] = masks.image_path.to_numpy()

    # Use is_good func to configure the label_index
    samples["label_index"] = samples["mask_path"].apply(is_mask_anomalous)
    samples.label_index = samples.label_index.astype(int)

    # Use label indexes to label data
    samples.loc[(samples.label_index == 0), "label"] = "Good"
    samples.loc[(samples.label_index == 1), "label"] = "Bad"

    # Add all 'Bad' samples to test set
    samples.loc[(samples.label == "Bad"), "split"] = "test"

    # Divide 'good' images to train/test on 0.8/0.2 ratio
    train_samples, test_samples = train_test_split(
        samples[samples.label == "Good"],
        train_size=train_split_ratio,
        random_state=42,
    )
    samples.loc[train_samples.index, "split"] = "train"
    samples.loc[test_samples.index, "split"] = "test"

    # Reorder columns
    samples = samples[["path", "item", "split", "label", "image_path", "mask_path", "label_index"]]

    # assert that the right mask files are associated with the right test images
    if not (
        samples.loc[samples.label_index == 1]
        .apply(lambda x: Path(x.image_path).stem in Path(x.mask_path).stem, axis=1)
        .all()
    ):
        msg = """Mismatch between anomalous images and ground truth masks. Make sure the mask files
        follow the same naming convention as the anomalous images in the dataset
        (e.g. image: 'Part0.jpg', mask: 'Part0_label.bmp')."""
        raise MisMatchError(msg)

    # Get the dataframe for the required split
    if split:
        samples = samples[samples.split == split].reset_index(drop=True)

    return samples


class KolektorDataset(AnomalibDataset):
    """Kolektor dataset class.

    Args:
        task (TaskType): Task type, ``classification``, ``detection`` or ``segmentation``
        root (Path | str): Path to the root of the dataset
            Defaults to ``./datasets/kolektor``.
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
        split (str | Split | None): Split of the dataset, usually Split.TRAIN or Split.TEST
            Defaults to ``None``.
    """

    def __init__(
        self,
        task: TaskType,
        root: Path | str = "./datasets/kolektor",
        transform: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(task=task, transform=transform)

        self.root = root
        self.split = split
        self.samples = make_kolektor_dataset(self.root, train_split_ratio=0.8, split=self.split)


class Kolektor(AnomalibDataModule):
    """Kolektor Datamodule.

    Args:
        root (Path | str): Path to the root of the dataset
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Test batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            Defaults to ``8``.
        task TaskType): Task type, 'classification', 'detection' or 'segmentation'
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
            Defaults to ``TestSplitMode.FROM_DIR``
        test_split_ratio (float): Fraction of images from the train set that will be reserved for testing.
            Defaults to ``0.2``
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
            Defaults to ``ValSplitMode.SAME_AS_TEST``
        val_split_ratio (float): Fraction of train or test images that will be reserved for validation.
            Defaults to ``0.5``
        seed (int | None, optional): Seed which may be set to a fixed value for reproducibility.
            Defaults to ``None``.
    """

    def __init__(
        self,
        root: Path | str = "./datasets/kolektor",
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

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = KolektorDataset(
            task=self.task,
            transform=self.train_transform,
            split=Split.TRAIN,
            root=self.root,
        )
        self.test_data = KolektorDataset(
            task=self.task,
            transform=self.eval_transform,
            split=Split.TEST,
            root=self.root,
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

                >> datamodule = Kolektor(root="./datasets/kolektor")
                >> datamodule.prepare_data()

            After:

            .. code-block:: bash

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
