"""Kolektor Surface-Defect Dataset.

Description:
    This script provides a PyTorch Dataset for the Kolektor
    Surface-Defect dataset. The dataset can be accessed at `Kolektor Surface-Defect Dataset <https://www.vicos.si/resources/kolektorsdd/>`_.

License:
    The Kolektor Surface-Defect dataset is released under the Creative Commons Attribution-NonCommercial-ShareAlike
    4.0 International License (CC BY-NC-SA 4.0). For more details, visit
    `Creative Commons License <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_.

Reference:
    Tabernik, Domen, Samo Šela, Jure Skvarč, and Danijel Skočaj. "Segmentation-based deep-learning approach
    for surface-defect detection." Journal of Intelligent Manufacturing 31, no. 3 (2020): 759-776.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
from cv2 import imread
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torchvision.transforms.v2 import Transform

from anomalib import TaskType
from anomalib.data.datasets import AnomalibDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import Split, validate_path

__all__ = ["KolektorDataset", "make_kolektor_dataset"]


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


def make_kolektor_dataset(
    root: str | Path,
    train_split_ratio: float = 0.8,
    split: str | Split | None = None,
) -> DataFrame:
    """Create Kolektor samples by parsing the Kolektor data file structure.

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

    # Get all image files and construct mask paths
    samples_data = []
    for image_path in root.glob("**/*.jpg"):
        mask_path = image_path.parent / f"{image_path.stem}_label.bmp"
        if not mask_path.exists():
            continue

        # Check if mask shows defects
        label_index = is_mask_anomalous(str(mask_path))
        label = "Bad" if label_index == 1 else "Good"

        # Determine split - all bad samples go to test
        split_type = "test" if label == "Bad" else None

        samples_data.append({
            "path": str(root),
            "item": image_path.parent.name,
            "split": split_type,
            "label": label,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "label_index": label_index,
        })

    if not samples_data:
        msg = f"Found 0 images in {root}"
        raise RuntimeError(msg)

    # Create DataFrame
    samples = DataFrame(samples_data)

    # Split good samples between train and test
    good_samples = samples[samples.label == "Good"]
    if not good_samples.empty:
        train_indices, test_indices = train_test_split(
            good_samples.index,
            train_size=train_split_ratio,
            random_state=42,
        )
        samples.loc[train_indices, "split"] = "train"
        samples.loc[test_indices, "split"] = "test"

    # Verify mask-image correspondence for anomalous samples
    anomalous_samples = samples[samples.label_index == 1]
    if not anomalous_samples.empty:
        mask_matches = anomalous_samples.apply(lambda x: Path(x.image_path).stem in Path(x.mask_path).stem, axis=1)
        if not mask_matches.all():
            msg = """Mismatch between anomalous images and ground truth masks. Make sure the mask files
            follow the same naming convention as the anomalous images in the dataset
            (e.g. image: 'Part0.jpg', mask: 'Part0_label.bmp')."""
            raise MisMatchError(msg)

    # Filter by split if specified
    if split:
        samples = samples[samples.split == split].reset_index(drop=True)

    return samples


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
