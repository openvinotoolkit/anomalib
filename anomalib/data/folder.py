"""Custom Folder Dataset.

This script creates a custom dataset from a folder.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Optional, Tuple, Union

import albumentations as A
from pandas import DataFrame
from torchvision.datasets.folder import IMG_EXTENSIONS

from anomalib.data.base import AnomalibDataModule, AnomalibDataset
from anomalib.data.utils import Split, ValSplitMode, random_split
from anomalib.pre_processing.pre_process import PreProcessor


def _check_and_convert_path(path: Union[str, Path]) -> Path:
    """Check an input path, and convert to Pathlib object.

    Args:
        path (Union[str, Path]): Input path.

    Returns:
        Path: Output path converted to pathlib object.
    """
    if not isinstance(path, Path):
        path = Path(path)
    return path


def _prepare_files_labels(
    path: Union[str, Path], path_type: str, extensions: Optional[Tuple[str, ...]] = None
) -> Tuple[list, list]:
    """Return a list of filenames and list corresponding labels.

    Args:
        path (Union[str, Path]): Path to the directory containing images.
        path_type (str): Type of images in the provided path ("normal", "abnormal", "normal_test")
        extensions (Optional[Tuple[str, ...]], optional): Type of the image extensions to read from the
            directory.

    Returns:
        List, List: Filenames of the images provided in the paths, labels of the images provided in the paths
    """
    path = _check_and_convert_path(path)
    if extensions is None:
        extensions = IMG_EXTENSIONS

    if isinstance(extensions, str):
        extensions = (extensions,)

    filenames = [f for f in path.glob(r"**/*") if f.suffix in extensions and not f.is_dir()]
    if len(filenames) == 0:
        raise RuntimeError(f"Found 0 {path_type} images in {path}")

    labels = [path_type] * len(filenames)

    return filenames, labels


def make_folder_dataset(
    normal_dir: Union[str, Path],
    abnormal_dir: Optional[Union[str, Path]] = None,
    normal_test_dir: Optional[Union[str, Path]] = None,
    mask_dir: Optional[Union[str, Path]] = None,
    split: Optional[Union[Split, str]] = None,
    extensions: Optional[Tuple[str, ...]] = None,
):
    """Make Folder Dataset.

    Args:
        normal_dir (Union[str, Path]): Path to the directory containing normal images.
        abnormal_dir (Optional[Union[str, Path]], optional): Path to the directory containing abnormal images.
        normal_test_dir (Optional[Union[str, Path]], optional): Path to the directory containing
            normal images for the test dataset. Normal test images will be a split of `normal_dir`
            if `None`. Defaults to None.
        mask_dir (Optional[Union[str, Path]], optional): Path to the directory containing
            the mask annotations. Defaults to None.
        split (Optional[Union[Split, str]], optional): Dataset split (ie., Split.FULL, Split.TRAIN or Split.TEST).
            Defaults to None.
        extensions (Optional[Tuple[str, ...]], optional): Type of the image extensions to read from the
            directory.

    Returns:
        DataFrame: an output dataframe containing samples for the requested split (ie., train or test)
    """

    filenames = []
    labels = []
    dirs = {"normal": normal_dir}

    if abnormal_dir:
        dirs = {**dirs, **{"abnormal": abnormal_dir}}

    if normal_test_dir:
        dirs = {**dirs, **{"normal_test": normal_test_dir}}

    for dir_type, path in dirs.items():
        filename, label = _prepare_files_labels(path, dir_type, extensions)
        filenames += filename
        labels += label

    samples = DataFrame({"image_path": filenames, "label": labels})

    # Create label index for normal (0) and abnormal (1) images.
    samples.loc[(samples.label == "normal") | (samples.label == "normal_test"), "label_index"] = 0
    samples.loc[(samples.label == "abnormal"), "label_index"] = 1
    samples.label_index = samples.label_index.astype(int)

    # If a path to mask is provided, add it to the sample dataframe.
    if mask_dir is not None:
        mask_dir = _check_and_convert_path(mask_dir)
        samples["mask_path"] = ""
        for index, row in samples.iterrows():
            if row.label_index == 1:
                samples.loc[index, "mask_path"] = str(mask_dir / row.image_path.name)

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


class FolderDataset(AnomalibDataset):
    """Folder dataset.

    Args:
        task (str): Task type. (classification or segmentation).
        pre_process (PreProcessor): Image Pre-processor to apply transform.
        split (Optional[Union[Split, str]]): Fixed subset split that follows from folder structure on file system.
            Choose from [Split.FULL, Split.TRAIN, Split.TEST]

        root (Union[str, Path]): Root folder of the dataset.
        normal_dir (Union[str, Path]): Path to the directory containing normal images.
        abnormal_dir (Optional[Union[str, Path]], optional): Path to the directory containing abnormal images.
        normal_test_dir (Optional[Union[str, Path]], optional): Path to the directory containing
            normal images for the test dataset. Defaults to None.
        mask_dir (Optional[Union[str, Path]], optional): Path to the directory containing
            the mask annotations. Defaults to None.

        extensions (Optional[Tuple[str, ...]], optional): Type of the image extensions to read from the
            directory.
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.

    Raises:
        ValueError: When task is set to classification and `mask_dir` is provided. When `mask_dir` is
            provided, `task` should be set to `segmentation`.
    """

    def __init__(
        self,
        task: str,
        pre_process: PreProcessor,
        root: Union[str, Path],
        normal_dir: Union[str, Path],
        abnormal_dir: Optional[Union[str, Path]] = None,
        normal_test_dir: Optional[Union[str, Path]] = None,
        mask_dir: Optional[Union[str, Path]] = None,
        split: Optional[Union[Split, str]] = None,
        val_split_mode: ValSplitMode = ValSplitMode.SAME_AS_TEST,
        extensions: Optional[Tuple[str, ...]] = None,
    ) -> None:
        super().__init__(task, pre_process)

        self.split = split
        self.normal_dir = Path(root) / Path(normal_dir)
        self.abnormal_dir = Path(root) / Path(abnormal_dir) if abnormal_dir else None
        self.normal_test_dir = Path(root) / Path(normal_test_dir) if normal_test_dir else None
        self.mask_dir = mask_dir
        self.extensions = extensions

        self.val_split_mode = val_split_mode

    def _setup(self):
        """Assign samples."""
        self.samples = make_folder_dataset(
            normal_dir=self.normal_dir,
            abnormal_dir=self.abnormal_dir,
            normal_test_dir=self.normal_test_dir,
            mask_dir=self.mask_dir,
            split=self.split,
            extensions=self.extensions,
        )


class Folder(AnomalibDataModule):
    """Folder DataModule.

    Args:
        root (Union[str, Path]): Path to the root folder containing normal and abnormal dirs.
        normal_dir (Union[str, Path]): Name of the directory containing normal images.
            Defaults to "normal".
        abnormal_dir (Union[str, Path]): Name of the directory containing abnormal images.
            Defaults to "abnormal".
        normal_test_dir (Optional[Union[str, Path]], optional): Path to the directory containing
            normal images for the test dataset. Defaults to None.
        mask_dir (Optional[Union[str, Path]], optional): Path to the directory containing
            the mask annotations. Defaults to None.
        split_ratio (float, optional): Ratio to split normal training images and add to the
            test set in case test set doesn't contain any normal images.
            Defaults to 0.2.
        extensions (Optional[Tuple[str, ...]], optional): Type of the image extensions to read from the
            directory. Defaults to None.
        image_size (Optional[Union[int, Tuple[int, int]]], optional): Size of the input image.
            Defaults to None.
        train_batch_size (int, optional): Training batch size. Defaults to 32.
        test_batch_size (int, optional): Test batch size. Defaults to 32.
        num_workers (int, optional): Number of workers. Defaults to 8.
        task (str, optional): Task type. Could be either classification or segmentation.
            Defaults to "classification".
        transform_config_train (Optional[Union[str, A.Compose]], optional): Config for pre-processing
            during training.
            Defaults to None.
        transform_config_val (Optional[Union[str, A.Compose]], optional): Config for pre-processing
            during validation.
            Defaults to None.
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
        seed (Optional[int], optional): Seed used during random subset splitting.
    """

    def __init__(
        self,
        root: Union[str, Path],
        normal_dir: Union[str, Path],
        abnormal_dir: Union[str, Path],
        normal_test_dir: Optional[Union[str, Path]] = None,
        mask_dir: Optional[Union[str, Path]] = None,
        split_ratio: float = 0.2,
        extensions: Optional[Tuple[str]] = None,
        #
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        task: str = "segmentation",
        transform_config_train: Optional[Union[str, A.Compose]] = None,
        transform_config_eval: Optional[Union[str, A.Compose]] = None,
        val_split_mode: ValSplitMode = ValSplitMode.FROM_TEST,
        seed: Optional[int] = None,
    ):
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            val_split_mode=val_split_mode,
            seed=seed,
        )

        self.split_ratio = split_ratio

        pre_process_train = PreProcessor(config=transform_config_train, image_size=image_size)
        pre_process_eval = PreProcessor(config=transform_config_eval, image_size=image_size)

        self.train_data = FolderDataset(
            task=task,
            pre_process=pre_process_train,
            split=Split.TRAIN,
            root=root,
            normal_dir=normal_dir,
            abnormal_dir=abnormal_dir,
            normal_test_dir=normal_test_dir,
            mask_dir=mask_dir,
            extensions=extensions,
        )

        self.test_data = FolderDataset(
            task=task,
            pre_process=pre_process_eval,
            split=Split.TEST,
            root=root,
            normal_dir=normal_dir,
            abnormal_dir=abnormal_dir,
            normal_test_dir=normal_test_dir,
            mask_dir=mask_dir,
            extensions=extensions,
        )

    def _setup(self, _stage: Optional[str] = None):
        """Set up the datasets for the Folder Data Module."""
        assert self.train_data is not None
        assert self.test_data is not None

        self.train_data.setup()
        self.test_data.setup()

        # add some normal images to the test set
        if not self.test_data.has_normal:
            self.train_data, normal_test_data = random_split(self.train_data, self.split_ratio, seed=self.seed)
            self.test_data += normal_test_data

        super()._setup()
