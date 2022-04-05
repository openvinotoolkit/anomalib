"""Custom Folder Dataset.

This script creates a custom dataset from a folder.
"""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
from pandas.core.frame import DataFrame
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS

from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import read_image
from anomalib.data.utils.split import (
    create_validation_set_from_test_set,
    split_normal_images_in_train_set,
)
from anomalib.pre_processing import PreProcessor

logger = logging.getLogger(name="Dataset: Folder Dataset")
logger.setLevel(logging.DEBUG)


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


def make_dataset(
    normal_dir: Union[str, Path],
    abnormal_dir: Union[str, Path],
    mask_dir: Optional[Union[str, Path]] = None,
    split: Optional[str] = None,
    split_ratio: float = 0.2,
    seed: int = 0,
    create_validation_set: bool = True,
    extensions: Optional[Tuple[str, ...]] = None,
):
    """Make Folder Dataset.

    Args:
        normal_dir (Union[str, Path]): Path to the directory containing normal images.
        abnormal_dir (Union[str, Path]): Path to the directory containing abnormal images.
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
    normal_dir = _check_and_convert_path(normal_dir)
    abnormal_dir = _check_and_convert_path(abnormal_dir)

    if extensions is None:
        extensions = IMG_EXTENSIONS

    # Get filenames from normal and abnormal directory.
    normal_filenames = [f for f in normal_dir.glob(r"**/*") if f.suffix in extensions]
    abnormal_filenames = [f for f in abnormal_dir.glob(r"**/*") if f.suffix in extensions]
    filenames = normal_filenames + abnormal_filenames

    if len(normal_filenames) == 0:
        raise RuntimeError(f"Found 0 normal images in {normal_dir}")

    if len(abnormal_filenames) == 0:
        raise RuntimeError(f"Found 0 annormal images in {abnormal_dir}")

    # Add normal and abnormal labels to the samples as `label` column.
    normal_labels = ["normal"] * len(normal_filenames)
    abnormal_labels = ["abnormal"] * len(abnormal_filenames)
    labels = normal_labels + abnormal_labels

    samples = DataFrame({"image_path": filenames, "label": labels})

    # Create label index for normal (0) and abnormal (1) images.
    samples.loc[(samples.label == "normal"), "label_index"] = 0
    samples.loc[(samples.label == "abnormal"), "label_index"] = 1
    samples.label_index = samples.label_index.astype(int)

    # If a path to mask is provided, add it to the sample dataframe.
    if mask_dir is not None:
        mask_dir = _check_and_convert_path(mask_dir)
        normal_gt = ["" for f in normal_filenames]
        abnormal_gt = [str(mask_dir / f.name) for f in abnormal_filenames]
        gt_filenames = normal_gt + abnormal_gt

        samples["mask_path"] = gt_filenames

    # Ensure the pathlib objects are converted to str.
    # This is because torch dataloader doesn't like pathlib.
    samples = samples.astype({"image_path": "str"})

    # Create train/test split.
    # By default, all the normal samples are assigned as train.
    #   and all the abnormal samples are test.
    samples.loc[(samples.label == "normal"), "split"] = "train"
    samples.loc[(samples.label == "abnormal"), "split"] = "test"
    samples = split_normal_images_in_train_set(
        samples=samples, split_ratio=split_ratio, seed=seed, normal_label="normal"
    )

    # If `create_validation_set` is set to True, the test set is split into half.
    if create_validation_set:
        samples = create_validation_set_from_test_set(samples, seed=seed, normal_label="normal")

    # Get the data frame for the split.
    if split is not None and split in ["train", "val", "test"]:
        samples = samples[samples.split == split]
        samples = samples.reset_index(drop=True)

    return samples


class FolderDataset(Dataset):
    """Folder Dataset."""

    def __init__(
        self,
        normal_dir: Union[Path, str],
        abnormal_dir: Union[Path, str],
        split: str,
        pre_process: PreProcessor,
        split_ratio: float = 0.2,
        mask_dir: Optional[Union[Path, str]] = None,
        extensions: Optional[Tuple[str, ...]] = None,
        task: Optional[str] = None,
        seed: int = 0,
        create_validation_set: bool = False,
    ) -> None:
        """Create Folder Folder Dataset.

        Args:
            normal_dir (Union[str, Path]): Path to the directory containing normal images.
            abnormal_dir (Union[str, Path]): Path to the directory containing abnormal images.
            split (Optional[str], optional): Dataset split (ie., either train or test). Defaults to None.
            pre_process (Optional[PreProcessor], optional): Image Pro-processor to apply transform.
                Defaults to None.
            split_ratio (float, optional): Ratio to split normal training images and add to the
                test set in case test set doesn't contain any normal images.
                Defaults to 0.2.
            mask_dir (Optional[Union[str, Path]], optional): Path to the directory containing
                the mask annotations. Defaults to None.
            extensions (Optional[Tuple[str, ...]], optional): Type of the image extensions to read from the
                directory.
            task (Optional[str], optional): Task type. (classification or segmentation) Defaults to None.
            seed (int, optional): Random seed to ensure reproducibility when splitting. Defaults to 0.
            create_validation_set (bool, optional):Boolean to create a validation set from the test set.
                Those wanting to create a validation set could set this flag to ``True``.

        Raises:
            ValueError: When task is set to classification and `mask_dir` is provided. When `mask_dir` is
                provided, `task` should be set to `segmentation`.

        """
        self.split = split

        if task == "classification" and mask_dir:
            raise ValueError(
                "Classification task is requested, but mask directory is provided. "
                "Segmentation task is to be chosen if mask directory is provided."
            )
        if task is None or mask_dir is None:
            self.task = "classification"
        else:
            self.task = task

        self.pre_process = pre_process
        self.samples = make_dataset(
            normal_dir=normal_dir,
            abnormal_dir=abnormal_dir,
            mask_dir=mask_dir,
            split=split,
            split_ratio=split_ratio,
            seed=seed,
            create_validation_set=create_validation_set,
            extensions=extensions,
        )

    def __len__(self) -> int:
        """Get length of the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Union[str, Tensor]]:
        """Get dataset item for the index ``index``.

        Args:
            index (int): Index to get the item.

        Returns:
            Union[Dict[str, Tensor], Dict[str, Union[str, Tensor]]]: Dict of image tensor during training.
                Otherwise, Dict containing image path, target path, image tensor, label and transformed bounding box.
        """
        item: Dict[str, Union[str, Tensor]] = {}

        image_path = self.samples.image_path[index]
        image = read_image(image_path)

        pre_processed = self.pre_process(image=image)
        item = {"image": pre_processed["image"]}

        if self.split in ["val", "test"]:
            label_index = self.samples.label_index[index]

            item["image_path"] = image_path
            item["label"] = label_index

            if self.task == "segmentation":
                mask_path = self.samples.mask_path[index]

                # Only Anomalous (1) images has masks in MVTec AD dataset.
                # Therefore, create empty mask for Normal (0) images.
                if label_index == 0:
                    mask = np.zeros(shape=image.shape[:2])
                else:
                    mask = cv2.imread(mask_path, flags=0) / 255.0

                pre_processed = self.pre_process(image=image, mask=mask)

                item["mask_path"] = mask_path
                item["image"] = pre_processed["image"]
                item["mask"] = pre_processed["mask"]

        return item


class FolderDataModule(LightningDataModule):
    """Folder Lightning Data Module."""

    def __init__(
        self,
        root: Union[str, Path],
        normal: str = "normal",
        abnormal: str = "abnormal",
        task: str = "classification",
        mask_dir: Optional[Union[Path, str]] = None,
        extensions: Optional[Tuple[str, ...]] = None,
        split_ratio: float = 0.2,
        seed: int = 0,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        train_batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 8,
        transform_config: Optional[Union[str, A.Compose]] = None,
        create_validation_set: bool = False,
    ) -> None:
        """Folder Dataset PL Datamodule.

        Args:
            root (Union[str, Path]): Path to the root folder containing normal and abnormal dirs.
            normal (str, optional): Name of the directory containing normal images.
                Defaults to "normal".
            abnormal (str, optional): Name of the directory containing abnormal images.
                Defaults to "abnormal".
            task (str, optional): Task type. Could be either classification or segmentation.
                Defaults to "classification".
            mask_dir (Optional[Union[str, Path]], optional): Path to the directory containing
                the mask annotations. Defaults to None.
            extensions (Optional[Tuple[str, ...]], optional): Type of the image extensions to read from the
                directory. Defaults to None.
            split_ratio (float, optional): Ratio to split normal training images and add to the
                test set in case test set doesn't contain any normal images.
                Defaults to 0.2.
            seed (int, optional): Random seed to ensure reproducibility when splitting. Defaults to 0.
            image_size (Optional[Union[int, Tuple[int, int]]], optional): Size of the input image.
                Defaults to None.
            train_batch_size (int, optional): Training batch size. Defaults to 32.
            test_batch_size (int, optional): Test batch size. Defaults to 32.
            num_workers (int, optional): Number of workers. Defaults to 8.
            transform_config (Optional[Union[str, A.Compose]], optional): Config for pre-processing.
                Defaults to None.
            create_validation_set (bool, optional):Boolean to create a validation set from the test set.
                Those wanting to create a validation set could set this flag to ``True``.

        Examples:
            Assume that we use Folder Dataset for the MVTec/bottle/broken_large category. We would do:
            >>> from anomalib.data import FolderDataModule
            >>> datamodule = FolderDataModule(
            ...     root="./datasets/MVTec/bottle/test",
            ...     normal="good",
            ...     abnormal="broken_large",
            ...     image_size=256
            ... )
            >>> datamodule.setup()
            >>> i, data = next(enumerate(datamodule.train_dataloader()))
            >>> data["image"].shape
            torch.Size([16, 3, 256, 256])

            >>> i, test_data = next(enumerate(datamodule.test_dataloader()))
            >>> test_data.keys()
            dict_keys(['image'])

            We could also create a Folder DataModule for datasets containing mask annotations.
            The dataset expects that mask annotation filenames must be same as the original filename.
            To this end, we modified mask filenames in MVTec AD bottle category.
            Now we could try folder data module using the mvtec bottle broken large category
            >>> datamodule = FolderDataModule(
            ...     root="./datasets/bottle/test",
            ...     normal="good",
            ...     abnormal="broken_large",
            ...     mask_dir="./datasets/bottle/ground_truth/broken_large",
            ...     image_size=256
            ... )

            >>> i , train_data = next(enumerate(datamodule.train_dataloader()))
            >>> train_data.keys()
            dict_keys(['image'])
            >>> train_data["image"].shape
            torch.Size([16, 3, 256, 256])

            >>> i, test_data = next(enumerate(datamodule.test_dataloader()))
            dict_keys(['image_path', 'label', 'mask_path', 'image', 'mask'])
            >>> print(test_data["image"].shape, test_data["mask"].shape)
            torch.Size([24, 3, 256, 256]) torch.Size([24, 256, 256])

            By default, Folder Data Module does not create a validation set. If a validation set
            is needed it could be set as follows:

            >>> datamodule = FolderDataModule(
            ...     root="./datasets/bottle/test",
            ...     normal="good",
            ...     abnormal="broken_large",
            ...     mask_dir="./datasets/bottle/ground_truth/broken_large",
            ...     image_size=256,
            ...     create_validation_set=True,
            ... )

            >>> i, val_data = next(enumerate(datamodule.val_dataloader()))
            >>> val_data.keys()
            dict_keys(['image_path', 'label', 'mask_path', 'image', 'mask'])
            >>> print(val_data["image"].shape, val_data["mask"].shape)
            torch.Size([12, 3, 256, 256]) torch.Size([12, 256, 256])

            >>> i, test_data = next(enumerate(datamodule.test_dataloader()))
            >>> print(test_data["image"].shape, test_data["mask"].shape)
            torch.Size([12, 3, 256, 256]) torch.Size([12, 256, 256])

        """
        super().__init__()

        self.root = _check_and_convert_path(root)
        self.normal_dir = self.root / normal
        self.abnormal_dir = self.root / abnormal
        self.mask_dir = mask_dir
        self.extensions = extensions
        self.split_ratio = split_ratio

        if task == "classification" and mask_dir is not None:
            raise ValueError(
                "Classification type is set but mask_dir provided. "
                "If mask_dir is provided task type must be segmentation. "
                "Check your configuration."
            )
        self.task = task
        self.transform_config = transform_config
        self.image_size = image_size

        self.pre_process = PreProcessor(config=self.transform_config, image_size=self.image_size)

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        self.create_validation_set = create_validation_set
        self.seed = seed

        self.train_data: Dataset
        self.test_data: Dataset
        if create_validation_set:
            self.val_data: Dataset
        self.inference_data: Dataset

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup train, validation and test data.

        Args:
          stage: Optional[str]:  Train/Val/Test stages. (Default value = None)

        """
        if stage in (None, "fit"):
            self.train_data = FolderDataset(
                normal_dir=self.normal_dir,
                abnormal_dir=self.abnormal_dir,
                split="train",
                split_ratio=self.split_ratio,
                mask_dir=self.mask_dir,
                pre_process=self.pre_process,
                extensions=self.extensions,
                task=self.task,
                seed=self.seed,
                create_validation_set=self.create_validation_set,
            )

        if self.create_validation_set:
            self.val_data = FolderDataset(
                normal_dir=self.normal_dir,
                abnormal_dir=self.abnormal_dir,
                split="val",
                split_ratio=self.split_ratio,
                mask_dir=self.mask_dir,
                pre_process=self.pre_process,
                extensions=self.extensions,
                task=self.task,
                seed=self.seed,
                create_validation_set=self.create_validation_set,
            )

        self.test_data = FolderDataset(
            normal_dir=self.normal_dir,
            abnormal_dir=self.abnormal_dir,
            split="test",
            split_ratio=self.split_ratio,
            mask_dir=self.mask_dir,
            pre_process=self.pre_process,
            extensions=self.extensions,
            task=self.task,
            seed=self.seed,
            create_validation_set=self.create_validation_set,
        )

        if stage == "predict":
            self.inference_data = InferenceDataset(
                path=self.root, image_size=self.image_size, transform_config=self.transform_config
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Get train dataloader."""
        return DataLoader(self.train_data, shuffle=True, batch_size=self.train_batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Get validation dataloader."""
        dataset = self.val_data if self.create_validation_set else self.test_data
        return DataLoader(dataset=dataset, shuffle=False, batch_size=self.test_batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Get test dataloader."""
        return DataLoader(self.test_data, shuffle=False, batch_size=self.test_batch_size, num_workers=self.num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        """Get predict dataloader."""
        return DataLoader(
            self.inference_data, shuffle=False, batch_size=self.test_batch_size, num_workers=self.num_workers
        )
