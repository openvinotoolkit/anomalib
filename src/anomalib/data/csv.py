"""Custom CSV Dataset.
This script creates a custom dataset from CSV files.
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import albumentations as A
from pandas import DataFrame

from anomalib.data.base import AnomalibDataModule, AnomalibDataset
from anomalib.data.task_type import TaskType
from anomalib.data.utils import (
    InputNormalizationMethod,
    Split,
    TestSplitMode,
    ValSplitMode,
    get_transforms,
)
from anomalib.data.utils.path import _prepare_files_labels_from_csv, _resolve_path


def make_csv_dataset(
    normal_csv: str | Path,
    root: str | Path | None = None,
    abnormal_csv: str | Path | None = None,
    normal_test_csv: str | Path | None = None,
    mask_csv: str | Path | None = None,
    split: str | Split | None = None,
    extensions: tuple[str, ...] | None = None,
) -> DataFrame:
    """Make Folder Dataset.
    Args:
        normal_csv (str | Path): Path to the CSV file containing normal images.
        root (str | Path | None): Path to the root directory of the dataset.
        abnormal_csv (str | Path | None, optional): Path to the CSV file containing abnormal images.
        normal_test_csv (str | Path | None, optional): Path to the CSV file containing
            normal images for the test dataset. Normal test images will be a split of `normal_csv`
            if `None`. Defaults to None.
        mask_csv (str | Path | None, optional): Path to the CSV file containing
            the mask annotations. Defaults to None.
        split (str | Split | None, optional): Dataset split (ie., Split.FULL, Split.TRAIN or Split.TEST).
            Defaults to None.
        extensions (tuple[str, ...] | None, optional): Type of the image extensions to read from the
            file.
        TODO: Need to define conventions for CSV file column names, or args for expected column names
    Returns:
        DataFrame: an output dataframe containing samples for the requested split (ie., train or test)
    """
    normal_csv = _resolve_path(normal_csv, root)
    abnormal_csv = _resolve_path(abnormal_csv, root) if abnormal_csv is not None else None
    normal_test_csv = _resolve_path(normal_test_csv, root) if normal_test_csv is not None else None
    mask_csv = _resolve_path(mask_csv, root) if mask_csv is not None else None
    assert normal_csv.is_file(), "A CSV file must be provided in normal_csv."

    filenames = []
    labels = []
    csvs = {"normal": normal_csv}

    if abnormal_csv:
        csvs = {**csvs, **{"abnormal": abnormal_csv}}

    if normal_test_csv:
        csvs = {**csvs, **{"normal_test": normal_test_csv}}

    if mask_csv:
        csvs = {**csvs, **{"mask_csv": mask_csv}}

    for csv_type, path in csvs.items():
        filename, label = _prepare_files_labels_from_csv(path, csv_type, extensions)
        filenames += filename
        labels += label

    samples = DataFrame({"image_path": filenames, "label": labels})
    samples = samples.sort_values(by="image_path", ignore_index=True)

    # Convert to absolute path if not
    samples.image_path = samples.image_path.apply(lambda path: _resolve_path(path, root))

    # Create label index for normal (0) and abnormal (1) images.
    samples.loc[(samples.label == "normal") | (samples.label == "normal_test"), "label_index"] = 0
    samples.loc[(samples.label == "abnormal"), "label_index"] = 1
    samples.label_index = samples.label_index.astype("Int64")

    # If a path to mask is provided, add it to the sample dataframe.

    if mask_csv is not None and abnormal_csv is not None:
        samples.loc[samples.label == "abnormal", "mask_path"] = samples.loc[
            samples.label == "mask_dir"
        ].image_path.values
        samples["mask_path"].fillna("", inplace=True)
        samples = samples.astype({"mask_path": "str"})

        # make sure all every rgb image has a corresponding mask image.
        assert (
            samples.loc[samples.label_index == 1]
            .apply(lambda x: Path(x.image_path).stem in Path(x.mask_path).stem, axis=1)
            .all()
        ), "Mismatch between anomalous images and mask images. Make sure the mask files \
            folder follow the same naming convention as the anomalous images in the dataset \
            (e.g. image: '000.png', mask: '000.png')."
    else:
        samples["mask_path"] = ""

    # remove all the rows with temporal image samples that have already been assigned
    samples = samples.loc[
        (samples.label == "normal") | (samples.label == "abnormal") | (samples.label == "normal_test")
    ]

    # Ensure the pathlib objects are converted to str.
    # This is because torch dataloader doesn't like pathlib.
    samples = samples.astype({"image_path": "str"})

    # Create train/test split.
    # By default, all the normal samples are assigned as train.
    #   and all the abnormal samples are test.
    samples.loc[(samples.label == "normal"), "split"] = "train"
    samples.loc[(samples.label == "abnormal") | (samples.label == "normal_test"), "split"] = "test"

    # Get the data frame for the split.
    if split:
        samples = samples[samples.split == split]
        samples = samples.reset_index(drop=True)

    return samples


class CSVDataset(AnomalibDataset):
    """CSV dataset.
    Args:
        task (TaskType): Task type. (``classification``, ``detection`` or ``segmentation``).
        transform (A.Compose): Albumentations Compose object describing the transforms that are applied to the inputs.
        split (str | Split | None): Fixed subset split that follows from folder structure on file system.
            Choose from [Split.FULL, Split.TRAIN, Split.TEST]
        normal_csv (str | Path): Path to the CSV file containing normal images.
        root (str | Path | None): Root folder of the dataset.
        abnormal_csv (str | Path | None, optional): Path to the CSV file containing abnormal images.
        normal_test_csv (str | Path | None, optional): Path to the CSV file containing
            normal images for the test dataset. Defaults to None.
        mask_csv (str | Path | None, optional): Path to the CSV file containing
            the mask annotations. Defaults to None.
        extensions (tuple[str, ...] | None, optional): Type of the image extensions to read from the
            directory.
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
    Raises:
        ValueError: When task is set to classification and `mask_csv` is provided. When `mask_csv` is
            provided, `task` should be set to `segmentation`.
    """

    def __init__(
        self,
        task: TaskType,
        transform: A.Compose,
        normal_csv: str | Path,
        root: str | Path | None = None,
        abnormal_csv: str | Path | None = None,
        normal_test_csv: str | Path | None = None,
        mask_csv: str | Path | None = None,
        split: str | Split | None = None,
        extensions: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(task, transform)

        self.split = split
        self.root = root
        self.normal_csv = normal_csv
        self.abnormal_csv = abnormal_csv
        self.normal_test_csv = normal_test_csv
        self.mask_csv = mask_csv
        self.extensions = extensions

    def _setup(self) -> None:
        """Assign samples."""
        self.samples = make_csv_dataset(
            root=self.root,
            normal_csv=self.normal_csv,
            abnormal_csv=self.abnormal_csv,
            normal_test_csv=self.normal_test_csv,
            mask_csv=self.mask_csv,
            split=self.split,
            extensions=self.extensions,
        )


class CSV(AnomalibDataModule):
    """CSV DataModule.
    Args:
        TODO: Update these
        normal_csv (str | Path): Name of the CSV file containing normal images.
            Defaults to "normal".
        root (str | Path | None): Path to the root folder containing normal and abnormal CSVs.
        abnormal_csv (str | Path | None): Name of the CSV file containing abnormal images.
            Defaults to "abnormal".
        normal_test_csv (str | Path | None, optional): Path to the CSV file containing
            normal images for the test dataset. Defaults to None.
        mask_csv (str | Path | None, optional): Path to the CSV file containing
            the mask annotations. Defaults to None.
        normal_split_ratio (float, optional): Ratio to split normal training images and add to the
            test set in case test set doesn't contain any normal images.
            Defaults to 0.2.
        extensions (tuple[str, ...] | None, optional): Type of the image extensions to read from the
            CSV. Defaults to None.
        image_size (int | tuple[int, int] | None, optional): Size of the input image.
            Defaults to None.
        center_crop (int | tuple[int, int] | None, optional): When provided, the images will be center-cropped
            to the provided dimensions.
        normalize (bool): When True, the images will be normalized to the ImageNet statistics.
        train_batch_size (int, optional): Training batch size. Defaults to 32.
        test_batch_size (int, optional): Test batch size. Defaults to 32.
        num_workers (int, optional): Number of workers. Defaults to 8.
        task (TaskType, optional): Task type. Could be ``classification``, ``detection`` or ``segmentation``.
            Defaults to segmentation.
        transform_config_train (str | A.Compose | None, optional): Config for pre-processing
            during training.
            Defaults to None.
        transform_config_val (str | A.Compose | None, optional): Config for pre-processing
            during validation.
            Defaults to None.
        test_split_mode (TestSplitMode): Setting that determines how the testing subset is obtained.
        test_split_ratio (float): Fraction of images from the train set that will be reserved for testing.
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
        val_split_ratio (float): Fraction of train or test images that will be reserved for validation.
        seed (int | None, optional): Seed used during random subset splitting.
    """

    def __init__(
        self,
        normal_csv: str | Path,
        root: str | Path | None = None,
        abnormal_csv: str | Path | None = None,
        normal_test_csv: str | Path | None = None,
        mask_csv: str | Path | None = None,
        normal_split_ratio: float = 0.2,
        extensions: tuple[str] | None = None,
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
        val_split_mode: ValSplitMode = ValSplitMode.FROM_TEST,
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

        self.normal_split_ratio = normal_split_ratio

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

        self.train_data = CSVDataset(
            task=task,
            transform=transform_train,
            split=Split.TRAIN,
            root=root,
            normal_csv=normal_csv,
            abnormal_csv=abnormal_csv,
            normal_test_csv=normal_test_csv,
            mask_csv=mask_csv,
            extensions=extensions,
        )

        self.test_data = CSVDataset(
            task=task,
            transform=transform_eval,
            split=Split.TEST,
            root=root,
            normal_csv=normal_csv,
            abnormal_csv=abnormal_csv,
            normal_test_csv=normal_test_csv,
            mask_csv=mask_csv,
            extensions=extensions,
        )
