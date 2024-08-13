"""Image CSV Dataset.

This script creates a custom dataset from a CSV file.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from torchvision.transforms.v2 import Transform

from anomalib import TaskType
from anomalib.data import AnomalibDataModule, AnomalibDataset
from anomalib.data.utils.label import LabelName
from anomalib.data.utils.split import SplitMode, resolve_split_mode

__all__ = ["CSV", "CSVDataset", "make_csv_dataset"]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names and data in the DataFrame.

    This function strips whitespace from column names and string data in the DataFrame.

    Args:
        df: Input DataFrame to be cleaned.

    Returns:
        DataFrame with cleaned column names and string data.

    Examples:
        >>> df = pd.DataFrame({'  Column1  ': ['  value1  ', '  value2  ']})
        >>> cleaned_df = clean_dataframe(df)
        >>> print(cleaned_df.columns)
        Index(['Column1'], dtype='object')
        >>> print(cleaned_df['Column1'].values)
        ['value1' 'value2']
    """
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
    return df


def map_labels(label: str) -> int:
    """Map label strings to LabelName enum values.

    Args:
        label: Input label string to be mapped.

    Returns:
        Integer representing the mapped label (0 for normal, 1 for abnormal, -1 for unmapped).

    Examples:
        >>> print(map_labels('normal'))
        0
        >>> print(map_labels('anomalous'))
        1
        >>> print(map_labels('unknown'))
        -1
    """
    label = label.lower()
    if label in ("normal", "good"):
        return LabelName.NORMAL
    if label in ("abnormal", "anomalous", "bad"):
        return LabelName.ABNORMAL
    return -1


def make_csv_dataset(
    csv_path: str | Path,
    split: Literal["train", "val", "test"] | None = None,
    sep: str | None = None,
    extension: str | list[str] | None = None,
    task: TaskType | str | None = None,
) -> pd.DataFrame:
    """Create a DataFrame from a CSV file containing image dataset information.

    This function reads a CSV file, cleans the data, maps labels, and optionally
    filters the dataset based on split and file extensions. It handles split
    assignment in the following order:
    1. Use the ``split`` column if it exists in the CSV.
    2. Use the split inferred from the filename (e.g., 'train.csv', 'valid.csv', 'test.csv').
    3. Use the split parameter if provided.
    4. If none of the above, assign splits based on normal/abnormal labels.

    Args:
        csv_path: Path to the CSV file containing image data.
        split: Dataset split to filter (i.e., 'train', 'valid', 'test', or None for all).
        sep: Delimiter to use for the CSV file. If None, will try to automatically detect.
        extension: File extension(s) to filter images. Can be a string for a
            single extension or a list of strings for multiple extensions.
        task: Task type for the dataset. Must be a valid TaskType enum value
            such as ``classification`` and ``segmentation``.

            Defaults to ``None``.

    Returns:
        DataFrame containing image samples, potentially filtered by split and/or extension.

    Raises:
        ValueError: If the CSV file doesn't contain required columns or if labels cannot be mapped.

    Examples:
        1. Load all data from a CSV file:

        >>> df = make_csv_dataset("path/to/data.csv")

        2. Load data from a CSV file with a specific split:

        >>> train_df = make_csv_dataset("path/to/data.csv", split="train")

        3. Load data from a CSV file named ``train.csv`` (split inferred from filename):

        >>> train_df = make_csv_dataset("path/to/train.csv")

        4. Load data and filter for specific image extensions:

        >>> jpg_png_df = make_csv_dataset("path/to/data.csv", extension=[".jpg", ".png"])

        5. Load data with a custom separator:

        >>> df = make_csv_dataset("path/to/data.csv", sep=";")

        6. Combine multiple options:

        >>> train_jpg_df = make_csv_dataset("path/to/data.csv", split="train", extension=".jpg")

        7. Load data from a CSV with an existing ``split`` column:

        >>> df = make_csv_dataset("path/to/data_with_splits.csv")
        # This will use the splits defined in the CSV file

        8. Load data without a ``split`` column or filename inference:

        >>> df = make_csv_dataset("path/to/unlabeled_data.csv")
        # This will assign 'train' to normal samples and 'test' to abnormal samples

    Note:
        - The function expects the CSV to have at least ``image_path`` and
            ``label`` columns.
        - If a ``mask_path`` column is present, it will be included in the output
            ``DataFrame``.
        - The function will automatically handle missing values in the
            ``mask_path`` column by replacing them with empty strings.
    """
    csv_path = Path(csv_path)
    samples = pd.read_csv(csv_path, sep=sep, engine="python" if sep is None else "c")

    required_columns = ["image_path", "label"]
    if not all(col in samples.columns for col in required_columns):
        msg = f"CSV must contain the following columns: {', '.join(required_columns)}"
        raise ValueError(msg)

    if task is not None:
        task = TaskType(task)
        if task == TaskType.SEGMENTATION and "mask_path" not in samples.columns:
            msg = "For segmentation tasks, the CSV file must contain a 'mask_path' column."
            raise ValueError(msg)

    samples = samples.sort_values(by="image_path", ignore_index=True)
    samples["label_index"] = samples["label"].map(map_labels).astype("int64")

    if (samples["label_index"] == -1).any():
        unmapped_labels = samples.loc[samples["label_index"] == -1, "label"].unique()
        msg = f"Unable to map the following labels: {unmapped_labels}"
        raise ValueError(msg)

    samples["image_path"] = samples["image_path"].astype(str)

    # Handle mask_path: if present, replace None or missing values with empty string
    if "mask_path" in samples.columns:
        samples["mask_path"] = samples["mask_path"].fillna("").astype(str)
    else:
        samples["mask_path"] = ""

    # Handle split assignment
    if "split" not in samples.columns:
        # Try to infer split from filename
        filename = csv_path.stem.lower()
        if filename in ["train", "valid", "test"]:
            samples["split"] = filename
        # Try to infer split from argument if provided
        elif split:
            samples["split"] = split
        else:
            # Assign splits based on normal/abnormal labels
            samples["split"] = np.where(samples["label_index"] == LabelName.NORMAL, "train", "test")

    # Filter by split if specified
    if split:
        samples = samples[samples["split"] == split]

    # Filter by extension if provided
    if extension:
        if isinstance(extension, str):
            extension = [extension]
        extension = [ext.lower() if not ext.startswith(".") else ext for ext in extension]
        samples = samples[samples["image_path"].str.lower().str.endswith(tuple(extension))]

    return samples.reset_index(drop=True)


class CSVDataset(AnomalibDataset):
    """Custom dataset class for image data loaded from a CSV file.

    This dataset class extends AnomalibDataset and provides methods to
    load data from a CSV file.

    Args:
        name (str): Name of the dataset. This is used to name the datamodule,
            especially when logging/saving.
        csv_path (str | Path): Path to the CSV file containing image data.
        task (str | TaskType): Task type for the dataset.
            Must be a valid TaskType enum value such as
            ``classification`` and ``segmentation``.
        split: Dataset split to filter (i.e., 'train', 'val', 'test').
            If it is None, no filtering is applied.

            Defaults to `None`.
        sep (str): Delimiter to use for the CSV file.
            If None, will try to automatically detect.
        extension (str): File extension(s) to filter images.
            Can be a string for a single extension or a list of strings for
            multiple extensions.
        transform (Transform | None): Optional transform to be applied to the data.

    Attributes:
        samples: DataFrame containing image samples, potentially filtered by
            split and/or extension.

    Examples:
        1. Create a dataset for a classification task with all data:

        >>> dataset = CSVDataset(
        ...     name="my_dataset",
        ...     csv_path="path/to/data.csv",
        ...     task="classification"
        ... )

        2. Create a dataset for a segmentation task, using only the training split:

        >>> train_dataset = CSVDataset(
        ...     name="segmentation_dataset",
        ...     csv_path="path/to/data.csv",
        ...     task="segmentation",
        ...     split="train"
        ... )

        3. Create a dataset with custom delimiter and specific file extensions:

        >>> custom_dataset = CSVDataset(
        ...     name="custom_dataset",
        ...     csv_path="path/to/data.csv",
        ...     task="classification",
        ...     sep=";",
        ...     extension=[".jpg", ".png"]
        ... )

        4. Create a dataset with a custom transform:

        >>> from torchvision import transforms
        >>> custom_transform = transforms.Compose([
        ...     transforms.Resize((224, 224)),
        ...     transforms.ToTensor(),
        ... ])
        >>> transformed_dataset = CSVDataset(
        ...     name="transformed_dataset",
        ...     csv_path="path/to/data.csv",
        ...     task="classification",
        ...     transform=custom_transform
        ... )

        5. Create datasets for different splits from the same CSV:

        >>> train_dataset = CSVDataset(name="train", csv_path="path/to/data.csv", task="classification", split="train")
        >>> val_dataset = CSVDataset(name="val", csv_path="path/to/data.csv", task="classification", split="val")
        >>> test_dataset = CSVDataset(name="test", csv_path="path/to/data.csv", task="classification", split="test")

    Note:
        - The CSV file should contain at least ``image_path`` and ``label`` columns.
        - If a ``mask_path`` column is present, it will be included for segmentation tasks.
        - The class uses the ``make_csv_dataset`` function internally to process the CSV file.
    """

    def __init__(
        self,
        name: str,
        csv_path: str | Path,
        task: str | TaskType,
        split: Literal["train", "val", "test"] | None = None,
        sep: str | None = None,
        extension: str | list[str] | None = None,
        transform: Transform | None = None,
    ) -> None:
        super().__init__(task, transform)
        self._name = name
        self.samples = make_csv_dataset(csv_path, split=split, sep=sep, extension=extension, task=task)

    @property
    def name(self) -> str:
        """Name of the dataset.

        Folder dataset overrides the name property to provide a custom name.
        """
        return self._name


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
        image_size (tuple[int, int] | None, optional): The size of the images.
            Defaults to ``None``.
        transform (Transform | None, optional): The transformations to apply to the dataset.
            Defaults to ``None``.
        train_transform (Transform | None, optional): The transformations to apply to the training dataset.
            Defaults to ``None``.
        eval_transform (Transform | None, optional): The transformations to apply to the evaluation dataset.
            Defaults to ``None``.
        test_split_mode (SplitMode | str, optional): The mode for splitting the test dataset.
            Defaults to ``SplitMode.PREDEFINED``.
        test_split_ratio (float, optional): The ratio for splitting the test dataset.
            Defaults to 0.2.
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
        This will load the train and test sets, and create a validatiation set
        from test set using a 50/50 split:

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
        ...     test_split_mode="predefined"   # or SplitMode.PREDEFINED,
        ...     val_split_mode="predefined"    # or SplitMode.PREDEFINED,
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
        This will load the train and test sets, and create a validatiation set
        from test set using a 50/50 split:

        >>> data_module = CSV(
        ...     name="segmentation_dataset",
        ...     csv_path="path/to/sample_dataset.csv",
        ...     task=TaskType.SEGMENTATION,
        ... )

        Custom Transforms
        -----------------
        Create a data module for segmentation with custom transforms:

        >>> from torchvision.transforms import v2 as T
        >>> transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
        >>> data_module = CSV(
        ...     name="custom_transform_dataset",
        ...     csv_path="path/to/sample_dataset.csv",
        ...     task=TaskType.SEGMENTATION,
        ...     transform=transform,
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
        image_size: tuple[int, int] | None = None,
        transform: Transform | None = None,
        train_transform: Transform | None = None,
        eval_transform: Transform | None = None,
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
            train_batch_size,
            eval_batch_size,
            num_workers,
            val_split_mode,
            val_split_ratio,
            test_split_mode,
            test_split_ratio,
            image_size,
            transform,
            train_transform,
            eval_transform,
            seed,
        )

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = CSVDataset(
            name="train",
            csv_path=self.csv_path,
            task=self.task,
            split="train",
            sep=self.sep,
            extension=self.extension,
            transform=self.train_transform,
        )

        if self.val_split_mode == SplitMode.PREDEFINED:
            self.val_data = CSVDataset(
                name="val",
                csv_path=self.csv_path,
                task=self.task,
                split="val",
                sep=self.sep,
                extension=self.extension,
                transform=self.eval_transform,
            )

        if self.test_split_mode == SplitMode.PREDEFINED:
            self.test_data = CSVDataset(
                name="test",
                csv_path=self.csv_path,
                task=self.task,
                split="test",
                sep=self.sep,
                extension=self.extension,
                transform=self.eval_transform,
            )

    @property
    def name(self) -> str:
        """Name of the datamodule.

        Folder datamodule overrides the name property to provide a custom name.
        """
        return self._name
