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
from anomalib.data.utils.path import _prepare_filemeta_from_csv, _resolve_path


def make_csv_dataset(
    csv_file: str | Path,
    root: str | Path | None = None,
    split: str | Split | None = None,
) -> DataFrame:
    """Make CSV Dataset.
    Args:
        csv_file (str | Path | None, optional): Path to the CSV file containing abnormal images.
        root (str | Path | None): Path to the root directory of the dataset.
        split (str | Split | None, optional): Dataset split (ie., Split.FULL, Split.TRAIN or Split.TEST).
            Defaults to None.
    Returns:
        DataFrame: an output dataframe containing samples for the requested split (ie., train or test)
    """
    csv_file = _resolve_path(csv_file, root)
    assert csv_file.is_file(), "A CSV file must be provided in csv_file."

    samples = _prepare_filemeta_from_csv(csv_file)
    samples = samples.sort_values(by="image_path", ignore_index=True)

    # Convert to absolute path if not
    samples.image_path = samples.image_path.apply(lambda path: _resolve_path(path, root))
    if "mask_path" in samples:
        samples.mask_path = samples.mask_path.apply(lambda path: _resolve_path(path, root))

    # Create label index for normal (0) and abnormal (1) images.
    if "label_index" not in samples:
        samples.loc[(samples.label == "normal") | (samples.label == "normal_test"), "label_index"] = 0
        samples.loc[(samples.label == "abnormal"), "label_index"] = 1
    samples.label_index = samples.label_index.astype("Int64")

    # If mask_path provided, validate
    if "mask_path" in samples:
        # make sure every rgb image has a corresponding mask image.
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
    if "split" not in samples:
        samples.loc[(samples.label == "normal"), "split"] = "train"
        samples.loc[(samples.label == "abnormal") | (samples.label == "normal_test"), "split"] = "test"
    else:
        # If split provided in csv, ensure train has only normal samples by excluding non-normal training samples
        samples = samples.loc[~((samples.label != "normal") & (samples.split == "train"))]

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
        split (str | Split | None): Fixed subset split. Choose from [Split.FULL, Split.TRAIN, Split.TEST]
        csv_file (str | Path): Path to the CSV file
        root (str | Path | None): Root folder of the dataset.
        extensions (tuple[str, ...] | None, optional): Type of the image extensions to read from the
            directory.
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
    """

    def __init__(
        self,
        task: TaskType,
        transform: A.Compose,
        csv_file: str | Path,
        root: str | Path | None = None,
        split: str | Split | None = None,
        extensions: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(task, transform)

        self.split = split
        self.root = root
        self.csv_file = csv_file
        self.extensions = extensions

    def _setup(self) -> None:
        """Assign samples."""
        self.samples = make_csv_dataset(csv_file=self.csv_file, root=self.root, split=self.split)


class CSV(AnomalibDataModule):
    """CSV DataModule.
    Args:
        csv_file (str | Path): Name of the CSV file containing images paths
        root (str | Path | None): Path to the root folder containing normal and abnormal CSVs.
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
        csv_file: str | Path,
        root: str | Path | None = None,
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
            csv_file=csv_file,
            extensions=extensions,
        )

        self.test_data = CSVDataset(
            task=task,
            transform=transform_eval,
            split=Split.TEST,
            root=root,
            csv_file=csv_file,
            extensions=extensions,
        )
