"""Dataloader for Datumaro format.

Note: This currently only works for annotations exported from Intel Geti™.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pandas as pd
from torchvision.transforms.v2 import Transform

from anomalib import TaskType
from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.utils import LabelName, Split


def make_datumaro_dataset(root: str | Path, split: str | Split | None = None) -> pd.DataFrame:
    """Make Datumaro Dataset.

    Assumes the following directory structure:

    dataset
    ├── annotations
    │    └── default.json
    └── images
        └── default
                ├── image1.jpg
                ├── image2.jpg
                └── ...

    Args:
        root (str | Path): Path to the dataset root directory.
        split (str | Split | None): Split of the dataset, usually Split.TRAIN or Split.TEST.
            Defaults to ``None``.

    Examples:
        >>> root = Path("path/to/dataset")
        >>> samples = make_datumaro_dataset(root)
        >>> samples.head()
            image_path	label	label_index	split	mask_path
        0	path/to/dataset...	Normal	0	Split.TRAIN
        1	path/to/dataset...	Normal	0	Split.TRAIN
        2	path/to/dataset...	Normal	0	Split.TRAIN
        3	path/to/dataset...	Normal	0	Split.TRAIN
        4	path/to/dataset...	Normal	0	Split.TRAIN


    Returns:
        DataFrame: an output dataframe containing samples for the requested split (ie., train or test).
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
            "mask_path": "",  # mask is provided in the annotation file and is not on disk.
        })
    samples_df = pd.DataFrame(
        samples,
        columns=["image_path", "label", "label_index", "split", "mask_path"],
        index=range(len(samples)),
    )
    # Create test/train split
    # By default assign all "Normal" samples to train and all "Anomalous" samples to test
    samples_df.loc[samples_df["label_index"] == LabelName.NORMAL, "split"] = Split.TRAIN
    samples_df.loc[samples_df["label_index"] == LabelName.ABNORMAL, "split"] = Split.TEST

    # Get the data frame for the split.
    if split:
        samples_df = samples_df[samples_df.split == split].reset_index(drop=True)

    return samples_df


class DatumaroDataset(AnomalibDataset):
    """Datumaro dataset class.

    Args:
        task (TaskType): Task type, ``classification``, ``detection`` or ``segmentation``.
        root (str | Path): Path to the dataset root directory.
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
        split (str | Split | None): Split of the dataset, usually Split.TRAIN or Split.TEST
            Defaults to ``None``.


    Examples:
        .. code-block:: python

            from anomalib.data.image.datumaro import DatumaroDataset
            from torchvision.transforms.v2 import Resize

            dataset = DatumaroDataset(root=root,
                task="classification",
                transform=Resize((256, 256)),
            )
            print(dataset[0].keys())
            # Output: dict_keys(['dm_format_version', 'infos', 'categories', 'items'])

    """

    def __init__(
        self,
        task: TaskType,
        root: str | Path,
        transform: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(task, transform)
        self.split = split
        self.samples = make_datumaro_dataset(root, split)
