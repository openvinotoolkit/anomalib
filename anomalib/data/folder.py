"""Custom Folder Dataset.

This script creates a custom dataset from a folder.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Optional, Tuple, Union

from pandas import DataFrame
from torchvision.datasets.folder import IMG_EXTENSIONS

from anomalib.data.base import AnomalibDataModule, AnomalibDataset, Split, ValSplitMode
from anomalib.data.utils.split import random_split
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
    abnormal_dir: Union[str, Path],
    normal_test_dir: Optional[Union[str, Path]] = None,
    mask_dir: Optional[Union[str, Path]] = None,
    split: Optional[str] = None,
    extensions: Optional[Tuple[str, ...]] = None,
):
    """Make Folder Dataset.

    Args:
        normal_dir (Union[str, Path]): Path to the directory containing normal images.
        abnormal_dir (Union[str, Path]): Path to the directory containing abnormal images.
        normal_test_dir (Optional[Union[str, Path]], optional): Path to the directory containing
            normal images for the test dataset. Normal test images will be a split of `normal_dir`
            if `None`. Defaults to None.
        mask_dir (Optional[Union[str, Path]], optional): Path to the directory containing
            the mask annotations. Defaults to None.
        split (Optional[str], optional): Dataset split (ie., either train or test). Defaults to None.
        split_ratio (float, optional): Ratio to split normal training images and add to the
            test set in case test set doesn't contain any normal images.
            Defaults to 0.2.
        seed (int, optional): Random seed to ensure reproducibility when splitting. Defaults to 0.
        create_validation_set (bool, optional):Boolean to create a validation set from the test set.
            Those wanting to create a validation set could set this flag to ``True``.
        extensions (Optional[Tuple[str, ...]], optional): Type of the image extensions to read from the
            directory.

    Returns:
        DataFrame: an output dataframe containing samples for the requested split (ie., train or test)
    """

    filenames = []
    labels = []
    dirs = {"normal": normal_dir, "abnormal": abnormal_dir}

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
    if split != Split.FULL:
        samples = samples[samples.split == split]
        samples = samples.reset_index(drop=True)

    return samples


class Folder(AnomalibDataset):
    def __init__(
        self,
        task: str,
        pre_process: PreProcessor,
        split: Split,
        #
        normal_dir: Union[str, Path],
        abnormal_dir: Union[str, Path],
        normal_test_dir: Optional[Union[str, Path]] = None,
        mask_dir: Optional[Union[str, Path]] = None,
        val_split_mode: ValSplitMode = ValSplitMode.SAME_AS_TEST,
        extensions=None,
        samples=None,
    ) -> None:
        super().__init__(task, pre_process, samples=samples)

        self.split = split

        self.normal_dir = normal_dir
        self.abnormal_dir = abnormal_dir
        self.normal_test_dir = normal_test_dir
        self.mask_dir = mask_dir
        self.extensions = extensions

        self.val_split_mode = val_split_mode

    def _setup(self):
        self._samples = make_folder_dataset(
            normal_dir=self.normal_dir,
            abnormal_dir=self.abnormal_dir,
            normal_test_dir=self.normal_test_dir,
            mask_dir=self.mask_dir,
            split=self.split,
            extensions=self.extensions,
        )


class FolderDataModule(AnomalibDataModule):
    def __init__(
        self,
        root,
        task,
        train_batch_size,
        test_batch_size,
        image_size,
        num_workers,
        val_split_mode,
        #
        normal_dir,
        abnormal_dir,
        normal_test_dir,
        mask_dir,
        split_ratio,
        transform_config_train=None,
        transform_config_val=None,
        extensions=None,
    ):
        super().__init__(
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
        )

        self.val_split_mode = val_split_mode
        self.split_ratio = split_ratio

        pre_process_train = PreProcessor(config=transform_config_train, image_size=image_size)
        pre_process_infer = PreProcessor(config=transform_config_val, image_size=image_size)

        normal_dir = Path(root) / Path(normal_dir)
        abnormal_dir = Path(root) / Path(abnormal_dir)

        self.train_data = Folder(
            task=task,
            pre_process=pre_process_train,
            split=Split.TRAIN,
            normal_dir=normal_dir,
            abnormal_dir=abnormal_dir,
            normal_test_dir=normal_test_dir,
            mask_dir=mask_dir,
            extensions=extensions,
        )

        self.test_data = Folder(
            task=task,
            pre_process=pre_process_infer,
            split=Split.TEST,
            normal_dir=normal_dir,
            abnormal_dir=abnormal_dir,
            normal_test_dir=normal_test_dir,
            mask_dir=mask_dir,
            extensions=extensions,
        )

    def _setup(self, _stage: Optional[str] = None):

        assert self.train_data is not None
        assert self.test_data is not None

        self.train_data.setup()
        self.test_data.setup()

        # add some normal images to the test set
        if not self.test_data.has_normal:
            self.train_data, normal_test_data = random_split(self.train_data, self.split_ratio)
            self.test_data += normal_test_data

        # split validation set from test set
        if self.val_split_mode == ValSplitMode.FROM_TEST:
            assert self.test_data is not None
            self.val_data, self.test_data = random_split(self.train_data, [0.5, 0.5], label_aware=True)
        elif self.val_split_mode == ValSplitMode.SAME_AS_TEST:
            self.val_data = self.test_data
        else:
            raise ValueError(f"Unknown validation split mode: {self.val_split_mode}")
