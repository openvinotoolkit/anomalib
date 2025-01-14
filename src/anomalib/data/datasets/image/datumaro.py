"""Dataloader for Datumaro format.

This module provides PyTorch Dataset implementation for loading images and
annotations in Datumaro format. Currently only supports annotations exported from
Intel Geti™.

The dataset expects the following directory structure::

    dataset/
    ├── annotations/
    │    └── default.json
    └── images/
        └── default/
                ├── image1.jpg
                ├── image2.jpg
                └── ...

The ``default.json`` file contains image paths and label annotations in Datumaro
format.

Example:
    >>> from pathlib import Path
    >>> from anomalib.data.datasets import DatumaroDataset
    >>> dataset = DatumaroDataset(
    ...     root=Path("./datasets/datumaro"),
    ...     split="train"
    ... )
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pandas as pd
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.utils import LabelName, Split


def make_datumaro_dataset(
    root: str | Path,
    split: str | Split | None = None,
) -> pd.DataFrame:
    """Create a DataFrame of image samples from a Datumaro dataset.

    Args:
        root (str | Path): Path to the dataset root directory.
        split (str | Split | None, optional): Dataset split to load. Usually
            ``Split.TRAIN`` or ``Split.TEST``. Defaults to ``None``.

    Returns:
        pd.DataFrame: DataFrame containing samples with columns:
            - ``image_path``: Path to the image file
            - ``label``: Class label name
            - ``label_index``: Numeric label index
            - ``split``: Dataset split
            - ``mask_path``: Path to mask file (empty for classification)

    Example:
        >>> root = Path("./datasets/datumaro")
        >>> samples = make_datumaro_dataset(root)
        >>> samples.head()  # doctest: +NORMALIZE_WHITESPACE
           image_path  label  label_index      split mask_path
        0  path/...   Normal           0  Split.TRAIN
        1  path/...   Normal           0  Split.TRAIN
        2  path/...   Normal           0  Split.TRAIN
    """
    annotation_file = Path(root) / "annotations" / "default.json"
    with annotation_file.open() as f:
        annotations = json.load(f)

    categories = annotations["categories"]
    categories = {idx: label["name"] for idx, label in enumerate(categories["label"]["labels"])}

    samples = []
    for item in annotations["items"]:
        image_path = Path(root) / "images" / "default" / item["image"]["path"]
        label_index = item["annotations"][0]["label_id"]
        label = categories[label_index]
        samples.append({
            "image_path": str(image_path),
            "label": label,
            "label_index": label_index,
            "split": None,
            "mask_path": "",  # mask is provided in annotation file
        })
    samples_df = pd.DataFrame(
        samples,
        columns=["image_path", "label", "label_index", "split", "mask_path"],
        index=range(len(samples)),
    )
    # Create test/train split
    # By default assign all "Normal" samples to train and all "Anomalous" to test
    samples_df.loc[samples_df["label_index"] == LabelName.NORMAL, "split"] = Split.TRAIN
    samples_df.loc[samples_df["label_index"] == LabelName.ABNORMAL, "split"] = Split.TEST

    # datumaro only supports classification
    samples_df.attrs["task"] = "classification"

    # Get the data frame for the split.
    if split:
        samples_df = samples_df[samples_df.split == split].reset_index(drop=True)

    return samples_df


class DatumaroDataset(AnomalibDataset):
    """Dataset class for loading Datumaro format datasets.

    Args:
        root (str | Path): Path to the dataset root directory.
        transform (Transform | None, optional): Transforms to apply to the images.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split to load. Usually
            ``Split.TRAIN`` or ``Split.TEST``. Defaults to ``None``.

    Example:
        >>> from pathlib import Path
        >>> from torchvision.transforms.v2 import Resize
        >>> from anomalib.data.datasets import DatumaroDataset
        >>> dataset = DatumaroDataset(
        ...     root=Path("./datasets/datumaro"),
        ...     transform=Resize((256, 256)),
        ...     split="train"
        ... )
    """

    def __init__(
        self,
        root: str | Path,
        augmentations: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)
        self.split = split
        self.samples = make_datumaro_dataset(root, split)
