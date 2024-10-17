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
from anomalib.data.base import AnomalibDataModule, AnomalibDataset
from anomalib.data.utils import LabelName, Split, TestSplitMode, ValSplitMode


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


class Datumaro(AnomalibDataModule):
    """Datumaro datamodule.

    Args:
        root (str | Path): Path to the dataset root directory.
        train_batch_size (int): Batch size for training dataloader.
            Defaults to ``32``.
        eval_batch_size (int): Batch size for evaluation dataloader.
            Defaults to ``32``.
        num_workers (int): Number of workers for dataloaders.
            Defaults to ``8``.
        task (TaskType): Task type, ``classification``, ``detection`` or ``segmentation``.
            Defaults to ``TaskType.CLASSIFICATION``. Currently only supports classification.
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
            Defaults to ``0.5``.
        seed (int | None, optional): Seed which may be set to a fixed value for reproducibility.
            Defualts to ``None``.

    Examples:
        To create a Datumaro datamodule

        >>> from pathlib import Path
        >>> from torchvision.transforms.v2 import Resize
        >>> root = Path("path/to/dataset")
        >>> datamodule = Datumaro(root, transform=Resize((256, 256)))
        >>> datamodule.setup()
        >>> i, data = next(enumerate(datamodule.train_dataloader()))
        >>> data.keys()
        dict_keys(['image_path', 'label', 'image'])

        >>> data["image"].shape
        torch.Size([32, 3, 256, 256])
    """

    def __init__(
        self,
        root: str | Path,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        task: TaskType = TaskType.CLASSIFICATION,
        image_size: tuple[int, int] | None = None,
        transform: Transform | None = None,
        train_transform: Transform | None = None,
        eval_transform: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.5,
        val_split_mode: ValSplitMode | str = ValSplitMode.FROM_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        if task != TaskType.CLASSIFICATION:
            msg = "Datumaro dataloader currently only supports classification task."
            raise ValueError(msg)
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            image_size=image_size,
            transform=transform,
            train_transform=train_transform,
            eval_transform=eval_transform,
            seed=seed,
        )
        self.root = root
        self.task = task

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = DatumaroDataset(
            task=self.task,
            root=self.root,
            transform=self.train_transform,
            split=Split.TRAIN,
        )
        self.test_data = DatumaroDataset(
            task=self.task,
            root=self.root,
            transform=self.eval_transform,
            split=Split.TEST,
        )
