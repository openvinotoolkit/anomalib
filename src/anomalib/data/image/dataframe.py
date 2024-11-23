"""Custom Dataframe Dataset.

This script creates a custom dataset from a pandas DataFrame.
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import IO

import pandas as pd
from torchvision.transforms.v2 import Transform

from anomalib import TaskType
from anomalib.data.base import AnomalibDataModule, AnomalibDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import (
    DirType,
    LabelName,
    Split,
    TestSplitMode,
    ValSplitMode,
)


def make_dataframe_dataset(
    samples: dict | list | pd.DataFrame,
    root: str | Path | None = None,
    split: str | Split | None = None,
) -> pd.DataFrame:
    """Make Folder Dataset.

    Args:
        samples (dict | list | pd.DataFrame): Pandas pd.DataFrame or compatible list or dict containing the
            dataset information.
        root (str | Path | None): Path to the root directory of the dataset.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split (ie., Split.FULL, Split.TRAIN or Split.TEST).
            Defaults to ``None``.

    Returns:
        pd.DataFrame: an output dataframe containing samples for the requested split (ie., train or test).

    Examples:
        Assume that we would like to use this ``make_dataframe_dataset`` to create a dataset from a pd.DataFrame.
        We could then create the dataset as follows,

        .. code-block:: python

            folder_df = make_dataframe_dataset(
                samples=input_df,
                split="train",
            )
            folder_df.head()

        .. code-block:: bash

                      image_path           label  label_index mask_path        split
            0  ./toy/good/00.jpg  DirType.NORMAL            0            Split.TRAIN
            1  ./toy/good/01.jpg  DirType.NORMAL            0            Split.TRAIN
            2  ./toy/good/02.jpg  DirType.NORMAL            0            Split.TRAIN
            3  ./toy/good/03.jpg  DirType.NORMAL            0            Split.TRAIN
            4  ./toy/good/04.jpg  DirType.NORMAL            0            Split.TRAIN
    """
    # Convert to pandas pd.DataFrame if dictionary or list is given
    if isinstance(samples, dict | list):
        samples = pd.DataFrame(samples)

    samples = samples.sort_values(by="image_path", ignore_index=True)

    # Create label column for folder datamodule compatibility
    samples.label_index = samples.label_index.astype("Int64")
    if "label" not in samples.columns:
        samples.loc[
            (samples.label_index == LabelName.NORMAL) & (samples.split == Split.TRAIN),
            "label",
        ] = DirType.NORMAL
        samples.loc[
            (samples.label_index == LabelName.NORMAL) & (samples.split == Split.TEST),
            "label",
        ] = DirType.NORMAL_TEST
        samples.loc[
            (samples.label_index == LabelName.ABNORMAL),
            "label",
        ] = DirType.ABNORMAL

    # Check if anomalous samples are in training set
    if len(samples[(samples.label_index == LabelName.ABNORMAL) & (samples.split == Split.TRAIN)]) != 0:
        msg = "Training set must not contain anomalous samples."
        raise MisMatchError(msg)

    # Add mask_path column if not exists
    if "mask_path" not in samples.columns:
        samples["mask_path"] = ""
    samples.loc[samples["mask_path"].isna(), "mask_path"] = ""

    # Add root to paths
    if root:
        samples["image_path"] = samples["image_path"].map(lambda x: Path(root, x))
        samples.loc[
            samples["mask_path"] != "",
            "mask_path",
        ] = samples.loc[samples["mask_path"] != "", "mask_path"].map(lambda x: Path(root, x))
    samples = samples.astype({"image_path": "str", "mask_path": "str", "label": "str"})

    # Get the dataframe for the split.
    if split:
        samples = samples[samples.split == split]
        samples = samples.reset_index(drop=True)

    return samples


class DataframeDataset(AnomalibDataset):
    """Dataframe dataset.

    This class is used to create a dataset from a pd.DataFrame. The class utilizes the Torch Dataset class.

    Args:
        name (str): Name of the dataset. This is used to name the datamodule, especially when logging/saving.
        task (TaskType): Task type. (``classification``, ``detection`` or ``segmentation``).
        samples (dict | list | pd.DataFrame): Pandas pd.DataFrame or compatible list or dict containing the
            dataset information.
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
        normal_dir (str | Path | Sequence): Path to the directory containing normal images.
        root (str | Path | None): Root folder of the dataset.
            Defaults to ``None``.
        split (str | Split | None): Fixed subset split that follows from folder structure on file system.
            Choose from [Split.FULL, Split.TRAIN, Split.TEST]
            Defaults to ``None``.

    Raises:
        ValueError: When task is set to classification and `mask_dir` is provided. When `mask_dir` is
            provided, `task` should be set to `segmentation`.

    Examples:
        Assume that we would like to use this ``DataframeDataset`` to create a dataset from a pd.DataFrame for
        a classification task. We could first create the transforms,

        >>> from anomalib.data.utils import InputNormalizationMethod, get_transforms
        >>> transform = get_transforms(image_size=256, normalization=InputNormalizationMethod.NONE)

        We could then create the dataset as follows,

        .. code-block:: python

            dataframe_dataset_classification_train = DataframeDataset(
                samples=input_df,
                split="train",
                transform=transform,
                task=TaskType.CLASSIFICATION,
            )

    """

    def __init__(
        self,
        name: str,
        task: TaskType,
        samples: dict | list | pd.DataFrame,
        transform: Transform | None = None,
        root: str | Path | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(task, transform)

        self._name = name
        self.root = root
        self.split = split
        self.samples = make_dataframe_dataset(
            samples=samples,
            root=self.root,
            split=self.split,
        )

    @property
    def name(self) -> str:
        """Name of the dataset.

        Dataframe dataset overrides the name property to provide a custom name.
        """
        return self._name


class Dataframe(AnomalibDataModule):
    """Dataframe DataModule.

    Args:
        name (str): Name of the dataset. This is used to name the datamodule, especially when logging/saving.
        samples (dict | list | pd.DataFrame): Pandas pd.DataFrame or compatible list or dict containing the
            dataset information.
        root (str | Path | None): Path to the root folder containing normal and abnormal dirs.
            Defaults to ``None``.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Validation, test and predict batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            Defaults to ``8``.
        task (TaskType, optional): Task type. Could be ``classification``, ``detection`` or ``segmentation``.
            Defaults to ``segmentation``.
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
            Defaults to ``ValSplitMode.FROM_TEST``.
        val_split_ratio (float): Fraction of train or test images that will be reserved for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Seed used during random subset splitting.
            Defaults to ``None``.

    Examples:
        The following code demonstrates how to use the ``Dataframe`` datamodule. Assume that the pandas pd.DataFrame
        ``input_df`` is structured as follows:

        .. code-block:: bash

                      image_path  label_index mask_path        split
            0  ./toy/good/00.jpg            0            Split.TRAIN
            1  ./toy/good/01.jpg            0            Split.TRAIN
            2  ./toy/good/02.jpg            0            Split.TRAIN
            3  ./toy/good/03.jpg            0            Split.TRAIN
            4  ./toy/good/04.jpg            0            Split.TRAIN

        .. code-block:: python

            dataframe_datamodule = Dataframe(
                "my_dataset",
                samples=input_df,
                root=dataset_root,
                task=TaskType.SEGMENTATION,
                image_size=256,
                normalization=InputNormalizationMethod.NONE,
            )
            dataframe_datamodule.setup()

        To access the training images,

        .. code-block:: python

            >> i, data = next(enumerate(dataframe_datamodule.train_dataloader()))
            >> print(data.keys(), data["image"].shape)

        To access the test images,

        .. code-block:: python

            >> i, data = next(enumerate(dataframe_datamodule.test_dataloader()))
            >> print(data.keys(), data["image"].shape)
    """

    def __init__(
        self,
        name: str,
        samples: dict | list | pd.DataFrame,
        root: str | Path | None = None,
        normal_split_ratio: float = 0.2,
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
        val_split_mode: ValSplitMode | str = ValSplitMode.FROM_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self._name = name
        self.root = root
        self._unprocessed_samples = samples
        self.task = TaskType(task)
        test_split_mode = TestSplitMode(test_split_mode)
        val_split_mode = ValSplitMode(val_split_mode)
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            image_size=image_size,
            transform=transform,
            train_transform=train_transform,
            eval_transform=eval_transform,
            seed=seed,
        )

        self.normal_split_ratio = normal_split_ratio

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = DataframeDataset(
            name=self.name,
            task=self.task,
            samples=self._unprocessed_samples,
            transform=self.train_transform,
            split=Split.TRAIN,
            root=self.root,
        )

        self.test_data = DataframeDataset(
            name=self.name,
            task=self.task,
            samples=self._unprocessed_samples,
            transform=self.eval_transform,
            split=Split.TEST,
            root=self.root,
        )

    @property
    def name(self) -> str:
        """Name of the datamodule.

        Dataframe datamodule overrides the name property to provide a custom name.
        """
        return self._name

    @classmethod
    def from_file(
        cls: type["Dataframe"],
        name: str,
        file_path: str | Path | IO[str] | IO[bytes],
        file_format: str = "csv",
        pd_kwargs: dict | None = None,
        **kwargs,
    ) -> "Dataframe":
        """Make Dataframe Datamodule from csv file.

        Args:
            name (str): Name of the dataset. This is used to name the datamodule,
                especially when logging/saving.
            file_path (str | Path | file-like): Path or file-like object to tabular
                file containing the datset information.
            file_format (str): File format supported by a pd.read_* method, such
                as ``csv``, ``parquet`` or ``json``.
                Defaults to ``csv``.
            pd_kwargs (dict | None): Keyword argument dictionary for the pd.read_* method.
                Defaults to ``None``.
            kwargs (dict): Additional keyword arguments for the Dataframe Datamodule class.

        Returns:
            Dataframe: Dataframe Datamodule
        """
        pd_kwargs = pd_kwargs or {}
        samples = getattr(pd, f"read_{file_format}")(file_path, **pd_kwargs)
        return cls(name, samples, **kwargs)
