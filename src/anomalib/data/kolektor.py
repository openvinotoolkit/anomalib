"""Kolektor Surface-Defect Dataset (CC BY-NC-SA 4.0).

Description:
    This script contains PyTorch Dataset, Dataloader and PyTorch
        Lightning DataModule for the Kolektor Surface-Defect dataset.

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
from imageio import imread
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from anomalib.data.base import AnomalibDataModule, AnomalibDataset
from anomalib.data.task_type import TaskType
from anomalib.data.utils import (
    InputNormalizationMethod,
    Split,
    TestSplitMode,
    ValSplitMode,
    get_transforms,
)

logger = logging.getLogger(__name__)


# Check if a mask shows defects
def is_good(path):
    img_arr = imread(path)
    if np.all(img_arr == 0):
        return True
    else:
        return False


def make_kolektor_dataset(
    root: str | Path,
    train_split_ratio: float = 0.8,
    split: str | Split | None = None,
) -> DataFrame:
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

    # Use is_good func to configure the label
    samples["label"] = samples["mask_path"].apply(is_good)
    samples.loc[(samples.label is True), "label"] = "Good"
    samples.loc[(samples.label is False), "label"] = "Bad"

    # Add label indexes
    samples.loc[(samples.label == "Good"), "label_index"] = 0
    samples.loc[(samples.label == "Bad"), "label_index"] = 1
    samples.label_index = samples.label_index.astype(int)

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
