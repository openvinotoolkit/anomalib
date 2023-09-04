"""Kolektor Surface-Defect Dataset (CC BY-NC-SA 4.0).

Description:
    This script contains PyTorch Dataset, Dataloader and PyTorch
        Lightning DataModule for the Kolektor Surface-Defect dataset.
    The dataset can be found at https://www.vicos.si/resources/kolektorsdd/

License:
    Kolektor Surface-Defect dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0)(https://creativecommons.org/licenses/by-nc-sa/4.0/).

Reference:
    - Tabernik, Domen, Samo Šela, Jure Skvarč, and Danijel Skočaj.
    "Segmentation-based deep-learning approach for surface-defect detection."
    Journal of Intelligent Manufacturing 31, no. 3 (2020): 759-776.
"""


from __future__ import annotations

import logging
from pathlib import Path

import albumentations as A
import numpy as np
from cv2 import imread
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from anomalib.data.base import AnomalibDataModule, AnomalibDataset
from anomalib.data.task_type import TaskType
from anomalib.data.utils import (
    DownloadInfo,
    InputNormalizationMethod,
    Split,
    TestSplitMode,
    ValSplitMode,
    download_and_extract,
    get_transforms,
)

logger = logging.getLogger(__name__)

DOWNLOAD_INFO = DownloadInfo(
    name="kolektor",
    url="https://go.vicos.si/kolektorsdd",
    hash="2b094030343c1cd59df02203ac6c57a0",
    filename="KolektorSDD.zip",
)


# Check if a mask shows defects
def is_mask_anomalous(path):
    img_arr = imread(path)
    if np.all(img_arr == 0):
        return 0
    return 1


def make_kolektor_dataset(
    root: str | Path,
    train_split_ratio: float = 0.8,
    split: str | Split | None = None,
) -> DataFrame:
    """Create Kolektor samples by parsing the Koelktor data file structure.

    The files are expected to follow the structure:
        image files:
            path/to/dataset/item/image_filename.jpg
            path/to/dataset/kos01/Part0.jpg
        mask files:
            path/to/dataset/item/mask_filename.bmp
            path/to/dataset/kos01/Part0_label.bmp

    This function creates a dataframe to store the parsed information based on the following format:
    |---|--------------------|--------|-------|---------|---------------------|--------------------|-------------|
    |   |    path            | item   | split | label   |  image_path         | mask_path          | label_index |
    |---|--------------------|--------|-------|---------|---------------------|--------------------|-------------|
    | 0 |   KolektorSDD      | kos01  | test  |  Bad    | /path/to/image_file | /path/to/mask_file |      1      |
    |---|--------------------|--------|-------|---------|---------------------|--------------------|-------------|

    Args:
        root (Path): Path to dataset
        train_split_ratio (float, optional): Ratio to split good images into train/test
            Defaults to 0.8 for train.
        split (str | Split | None, optional): Dataset split (Either train or test). Defaults to None.

    Examples:
        The following example shows how to get training samples from Kolektor Dataset:

        >>> root = Path('./KolektorSDD/')

        >>> samples = make_kolektor_dataset(root, train_split_ratio=0.8)
        >>> print(samples.head())
                path      item   split  label        image_path                        mask_path             label_index
        0    KolektorSDD  kos01  train  Good  KolektorSDD/kos01/Part0.jpg  KolektorSDD/kos01/Part0_label.bmp      0
        1    KolektorSDD  kos01  train  Good  KolektorSDD/kos01/Part1.jpg  KolektorSDD/kos01/Part1_label.bmp      0
        2    KolektorSDD  kos01  train  Good  KolektorSDD/kos01/Part2.jpg  KolektorSDD/kos01/Part2_label.bmp      0
        3    KolektorSDD  kos01   test  Good  KolektorSDD/kos01/Part3.jpg  KolektorSDD/kos01/Part3_label.bmp      0
        4    KolektorSDD  kos01  train  Good  KolektorSDD/kos01/Part4.jpg  KolektorSDD/kos01/Part4_label.bmp      0

    Returns:
        DataFrame: an output dataframe containing the samples of the dataset.
    """

    root = Path(root)

    # Get list of images and masks
    samples_list = [(str(root),) + f.parts[-2:] for f in root.glob(r"**/*") if f.suffix == ".jpg"]
    masks_list = [(str(root),) + f.parts[-2:] for f in root.glob(r"**/*") if f.suffix == ".bmp"]

    if not samples_list:
        raise RuntimeError(f"Found 0 images in {root}")

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
    samples["mask_path"] = masks.image_path.values

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
        samples[samples.label == "Good"], train_size=train_split_ratio, random_state=42
    )
    samples.loc[train_samples.index, "split"] = "train"
    samples.loc[test_samples.index, "split"] = "test"

    # Reorder columns
    samples = samples[["path", "item", "split", "label", "image_path", "mask_path", "label_index"]]

    # assert that the right mask files are associated with the right test images
    assert (
        samples.loc[samples.label_index == 1]
        .apply(lambda x: Path(x.image_path).stem in Path(x.mask_path).stem, axis=1)
        .all()
    ), "Mismatch between anomalous images and ground truth masks. Make sure the mask files  \
        follow the same naming convention as the anomalous images in the dataset (e.g. image: 'Part0.jpg', \
        mask: 'Part0_label.bmp')."

    # Get the dataframe for the required split
    if split:
        samples = samples[samples.split == split].reset_index(drop=True)

    return samples


class KolektorDataset(AnomalibDataset):
    """Kolektor dataset class.

    Args:
        task (TaskType): Task type, ``classification``, ``detection`` or ``segmentation``
        transform (A.Compose): Albumentations Compose object describing the transforms that are applied to the inputs.
        root (Path | str): Path to the root of the dataset
        split (str | Split | None): Split of the dataset, usually Split.TRAIN or Split.TEST
    """

    def __init__(
        self,
        task: TaskType,
        transform: A.Compose,
        root: Path | str,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(task=task, transform=transform)

        self.root = root
        self.split = split

    def _setup(self) -> None:
        self.samples = make_kolektor_dataset(self.root, train_split_ratio=0.8, split=self.split)


class Kolektor(AnomalibDataModule):
    """Kolektor Datamodule.

    Args:
        root (Path | str): Path to the root of the dataset
        image_size (int | tuple[int, int] | None, optional): Size of the input image.
            Defaults to None.
        center_crop (int | tuple[int, int] | None, optional): When provided, the images will be center-cropped
            to the provided dimensions.
        normalize (bool): When True, the images will be normalized to the ImageNet statistics.
        train_batch_size (int, optional): Training batch size. Defaults to 32.
        eval_batch_size (int, optional): Test batch size. Defaults to 32.
        num_workers (int, optional): Number of workers. Defaults to 8.
        task TaskType): Task type, 'classification', 'detection' or 'segmentation'
        transform_config_train (str | A.Compose | None, optional): Config for pre-processing
            during training.
            Defaults to None.
        transform_config_val (str | A.Compose | None, optional): Config for pre-processing
            during validation.
            Defaults to None.
        test_split_mode (TestSplitMode): Setting that determines how the testing subset is obtained.
        test_split_ratio (float): Fraction of images from the train set that will be reserved for testing.
            Defaults to 0.2
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
        val_split_ratio (float): Fraction of train or test images that will be reserved for validation.
            Defaults to 0.5
        seed (int | None, optional): Seed which may be set to a fixed value for reproducibility.
    """

    def __init__(
        self,
        root: Path | str,
        image_size: int | tuple[int, int] | None = None,
        center_crop: int | tuple[int, int] | None = None,
        normalization: str | InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        task: TaskType = TaskType.SEGMENTATION,
        transform_config_train: str | A.Compose | None = None,
        transform_config_eval: str | A.Compose | None = None,
        test_split_mode: TestSplitMode = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode = ValSplitMode.SAME_AS_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        self.root = Path(root)

        transform_train = get_transforms(
            config=transform_config_train,
            image_size=image_size,
            center_crop=center_crop,
            normalization=InputNormalizationMethod(normalization),
        )
        transform_eval = get_transforms(
            config=transform_config_eval,
            image_size=image_size,
            center_crop=center_crop,
            normalization=InputNormalizationMethod(normalization),
        )

        self.train_data = KolektorDataset(
            task=task,
            transform=transform_train,
            split=Split.TRAIN,
            root=root,
        )
        self.test_data = KolektorDataset(
            task=task,
            transform=transform_eval,
            split=Split.TEST,
            root=root,
        )

    def prepare_data(self) -> None:
        """Download the dataset if not available."""
        if (self.root).is_dir():
            logger.info("Found the dataset.")
        else:
            download_and_extract(self.root, DOWNLOAD_INFO)
