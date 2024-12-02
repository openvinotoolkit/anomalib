"""Image CSV Datamodule.

This script creates a custom dataset from a CSV file.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Literal

from anomalib import TaskType
from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.csv import CSVDataset
from anomalib.data.utils.split import SplitMode, resolve_split_mode


class CSV(AnomalibDataModule):
    """A data module class for loading data from a CSV file.

    This class is designed to handle datasets for various tasks such as classification and segmentation.
    It reads the data from a CSV file and processes it accordingly, providing data loaders for training,
    validation, and testing.

    Args:
        name (str): The name of the dataset.
        csv_path (str | Path): The path to the CSV file containing the dataset.
        split (Literal["train", "val", "test"] | None, optional): The dataset split to use.
            Defaults to ``None``.
        sep (str | None, optional): The delimiter to use for the CSV file.
            Defaults to ``None``.
        extension (str | list[str] | None, optional): The file extension(s) to filter the dataset.
            Defaults to ``None``.
        task (TaskType | str, optional): The type of task (e.g., "classification", "segmentation").
            Defaults to ``TaskType.SEGMENTATION``.
        train_batch_size (int, optional): The batch size for training.
            Defaults to ``32``.
        eval_batch_size (int, optional): The batch size for evaluation.
            Defaults to ``32``.
        num_workers (int, optional): The number of workers for data loading.
            Defaults to ``8``.
        test_split_mode (SplitMode | str, optional): The mode for splitting the test dataset.
            Defaults to ``SplitMode.AUTO``.
        test_split_ratio (float, optional): The ratio for splitting the test dataset.
            Defaults to 0.4.
        val_split_mode (SplitMode | str, optional): The mode for splitting the validation dataset.
            Defaults to ``SplitMode.AUTO``.
        val_split_ratio (float, optional): The ratio for splitting the validation dataset.
            Defaults to ``0.5``.
        seed (int | None, optional): The random seed for splitting the dataset.
            Defaults to ``None``.

    Examples:
        Assume we have a CSV file named 'sample_dataset.csv' with the following structure:

        .. csv-table::
           :header: "image_path", "label", "mask_path"
           :widths: 20, 10, 20

           "images/normal_1.jpg", "normal", ""
           "images/normal_2.png", "normal", ""
           "images/abnormal_1.jpg", "abnormal", "masks/abnormal_1_mask.png"
           "images/abnormal_2.png", "abnormal", "masks/abnormal_2_mask.png"

        Classification Tasks
        ====================

        Basic Usage
        -----------
        Create a data module for a classification task with default settings.
        This will load the train and test sets, and create a validation set
        from the test set using a 50/50 split:

        >>> data_module = CSV(
        ...     name="classification_dataset",
        ...     csv_path="path/to/sample_dataset.csv",
        ...     task="classification"   # or TaskType.CLASSIFICATION,
        ... )

        Custom Splits
        -------------
        Create a data module with custom split modes for classification:

        >>> data_module = CSV(
        ...     name="custom_split_dataset",
        ...     csv_path="path/to/sample_dataset.csv",
        ...     task=TaskType.CLASSIFICATION,
        ...     test_split_mode=SplitMode.PREDEFINED,
        ...     val_split_mode=SplitMode.PREDEFINED,
        ... )

        Custom Format
        -------------
        Create a data module with a custom separator and file extension filter:

        >>> data_module = CSV(
        ...     name="custom_format_dataset",
        ...     csv_path="path/to/sample_dataset.csv",
        ...     task=TaskType.CLASSIFICATION,
        ...     sep=";",
        ...     extension=[".jpg", ".png"],
        ... )

        Segmentation Tasks
        ==================

        Basic Usage
        -----------
        Create a data module for a segmentation task with default settings:
        This will load the train and test sets, and create a validation set
        from the test set using a 50/50 split:

        >>> data_module = CSV(
        ...     name="segmentation_dataset",
        ...     csv_path="path/to/sample_dataset.csv",
        ...     task=TaskType.SEGMENTATION,
        ... )

        Predefined Splits
        -----------------
        Create a data module with predefined splits (assuming the CSV has a 'split' column):
        This will load the train, validation, and test sets based on the 'split' column:

        >>> data_module = CSV(
        ...     name="predefined_split_dataset",
        ...     csv_path="path/to/sample_dataset_with_splits.csv",
        ...     task=TaskType.SEGMENTATION,
        ...     test_split_mode=SplitMode.PREDEFINED,
        ...     val_split_mode=SplitMode.PREDEFINED,
        ... )

    Note:
        - The CSV file should contain at least 'image_path' and 'label' columns.
        - For segmentation tasks, a 'mask_path' column is required.
        - The 'split' column in the CSV file, if present, should contain 'train', 'val', or 'test' values.
        - When using SplitMode.PREDEFINED, ensure your CSV file contains a 'split' column.
        - SplitMode.SYNTHETIC is used to generate synthetic data for splitting when real data is unavailable.
        - SplitMode.AUTO automatically determines the best splitting strategy based on available data.
    """

    def __init__(
        self,
        name: str,
        csv_path: str | Path,
        split: Literal["train", "val", "test"] | None = None,
        sep: str | None = None,
        extension: str | list[str] | None = None,
        task: TaskType | str = TaskType.SEGMENTATION,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        test_split_mode: SplitMode | str = SplitMode.AUTO,
        test_split_ratio: float = 0.4,
        val_split_mode: SplitMode | str = SplitMode.AUTO,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self._name = name
        self.csv_path = csv_path
        self.split = split
        self.sep = sep
        self.task = TaskType(task)
        self.extension = extension
        val_split_mode = resolve_split_mode(val_split_mode)
        test_split_mode = resolve_split_mode(test_split_mode)
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            seed=seed,
        )

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = CSVDataset(
            name="train",
            csv_path=self.csv_path,
            task=self.task,
            split="train",
            sep=self.sep,
            extension=self.extension,
        )

        if self.val_split_mode == SplitMode.PREDEFINED:
            self.val_data = CSVDataset(
                name="val",
                csv_path=self.csv_path,
                task=self.task,
                split="val",
                sep=self.sep,
                extension=self.extension,
            )

        if self.test_split_mode == SplitMode.PREDEFINED:
            self.test_data = CSVDataset(
                name="test",
                csv_path=self.csv_path,
                task=self.task,
                split="test",
                sep=self.sep,
                extension=self.extension,
            )

    @property
    def name(self) -> str:
        """Name of the datamodule.

        Folder datamodule overrides the name property to provide a custom name.
        """
        return self._name
