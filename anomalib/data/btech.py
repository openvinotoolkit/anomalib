"""BTech Dataset.

This script contains PyTorch Lightning DataModule for the BTech dataset.

If the dataset is not on the file system, the script downloads and
extracts the dataset and create PyTorch data objects.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import shutil
import warnings
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from urllib.request import urlretrieve

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import VisionDataset
from tqdm import tqdm

from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import DownloadProgressBar, hash_check, read_image
from anomalib.data.utils.split import (
    create_validation_set_from_test_set,
    split_normal_images_in_train_set,
)
from anomalib.pre_processing import PreProcessor

logger = logging.getLogger(__name__)


def make_btech_dataset(
    path: Path,
    split: Optional[str] = None,
    split_ratio: float = 0.1,
    seed: Optional[int] = None,
    create_validation_set: bool = False,
) -> DataFrame:
    """Create BTech samples by parsing the BTech data file structure.

    The files are expected to follow the structure:
        path/to/dataset/split/category/image_filename.png
        path/to/dataset/ground_truth/category/mask_filename.png

    Args:
        path (Path): Path to dataset
        split (str, optional): Dataset split (ie., either train or test). Defaults to None.
        split_ratio (float, optional): Ratio to split normal training images and add to the
            test set in case test set doesn't contain any normal images.
            Defaults to 0.1.
        seed (int, optional): Random seed to ensure reproducibility when splitting. Defaults to 0.
        create_validation_set (bool, optional): Boolean to create a validation set from the test set.
            BTech dataset does not contain a validation set. Those wanting to create a validation set
            could set this flag to ``True``.

    Example:
        The following example shows how to get training samples from BTech 01 category:

        >>> root = Path('./BTech')
        >>> category = '01'
        >>> path = root / category
        >>> path
        PosixPath('BTech/01')

        >>> samples = make_btech_dataset(path, split='train', split_ratio=0.1, seed=0)
        >>> samples.head()
           path     split label image_path                  mask_path                     label_index
        0  BTech/01 train 01    BTech/01/train/ok/105.bmp BTech/01/ground_truth/ok/105.png      0
        1  BTech/01 train 01    BTech/01/train/ok/017.bmp BTech/01/ground_truth/ok/017.png      0
        ...

    Returns:
        DataFrame: an output dataframe containing samples for the requested split (ie., train or test)
    """
    samples_list = [
        (str(path),) + filename.parts[-3:] for filename in path.glob("**/*") if filename.suffix in (".bmp", ".png")
    ]
    if len(samples_list) == 0:
        raise RuntimeError(f"Found 0 images in {path}")

    samples = pd.DataFrame(samples_list, columns=["path", "split", "label", "image_path"])
    samples = samples[samples.split != "ground_truth"]

    # Create mask_path column
    samples["mask_path"] = (
        samples.path
        + "/ground_truth/"
        + samples.label
        + "/"
        + samples.image_path.str.rstrip("png").str.rstrip(".")
        + ".png"
    )

    # Modify image_path column by converting to absolute path
    samples["image_path"] = samples.path + "/" + samples.split + "/" + samples.label + "/" + samples.image_path

    # Split the normal images in training set if test set doesn't
    # contain any normal images. This is needed because AUC score
    # cannot be computed based on 1-class
    if sum((samples.split == "test") & (samples.label == "ok")) == 0:
        samples = split_normal_images_in_train_set(samples, split_ratio, seed)

    # Good images don't have mask
    samples.loc[(samples.split == "test") & (samples.label == "ok"), "mask_path"] = ""

    # Create label index for normal (0) and anomalous (1) images.
    samples.loc[(samples.label == "ok"), "label_index"] = 0
    samples.loc[(samples.label != "ok"), "label_index"] = 1
    samples.label_index = samples.label_index.astype(int)

    if create_validation_set:
        samples = create_validation_set_from_test_set(samples, seed=seed)

    # Get the data frame for the split.
    if split is not None and split in ["train", "val", "test"]:
        samples = samples[samples.split == split]
        samples = samples.reset_index(drop=True)

    return samples


class BTechDataset(VisionDataset):
    """BTech PyTorch Dataset."""

    def __init__(
        self,
        root: Union[Path, str],
        category: str,
        pre_process: PreProcessor,
        split: str,
        task: str = "segmentation",
        seed: Optional[int] = None,
        create_validation_set: bool = False,
    ) -> None:
        """Btech Dataset class.

        Args:
            root: Path to the BTech dataset
            category: Name of the BTech category.
            pre_process: List of pre_processing object containing albumentation compose.
            split: 'train', 'val' or 'test'
            task: ``classification`` or ``segmentation``
            seed: seed used for the random subset splitting
            create_validation_set: Create a validation subset in addition to the train and test subsets

        Examples:
            >>> from anomalib.data.btech import BTechDataset
            >>> from anomalib.data.transforms import PreProcessor
            >>> pre_process = PreProcessor(image_size=256)
            >>> dataset = BTechDataset(
            ...     root='./datasets/BTech',
            ...     category='leather',
            ...     pre_process=pre_process,
            ...     task="classification",
            ...     is_train=True,
            ... )
            >>> dataset[0].keys()
            dict_keys(['image'])

            >>> dataset.split = "test"
            >>> dataset[0].keys()
            dict_keys(['image', 'image_path', 'label'])

            >>> dataset.task = "segmentation"
            >>> dataset.split = "train"
            >>> dataset[0].keys()
            dict_keys(['image'])

            >>> dataset.split = "test"
            >>> dataset[0].keys()
            dict_keys(['image_path', 'label', 'mask_path', 'image', 'mask'])

            >>> dataset[0]["image"].shape, dataset[0]["mask"].shape
            (torch.Size([3, 256, 256]), torch.Size([256, 256]))
        """
        super().__init__(root)

        if seed is None:
            warnings.warn(
                "seed is None."
                " When seed is not set, images from the normal directory are split between training and test dir."
                " This will lead to inconsistency between runs."
            )

        self.root = Path(root) if isinstance(root, str) else root
        self.category: str = category
        self.split = split
        self.task = task

        self.pre_process = pre_process

        self.samples = make_btech_dataset(
            path=self.root / category,
            split=self.split,
            seed=seed,
            create_validation_set=create_validation_set,
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

                # Only Anomalous (1) images has masks in BTech dataset.
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


@DATAMODULE_REGISTRY
class BTech(LightningDataModule):
    """BTechDataModule Lightning Data Module."""

    def __init__(
        self,
        root: str,
        category: str,
        # TODO: Remove default values. IAAALD-211
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        train_batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 8,
        task: str = "segmentation",
        transform_config_train: Optional[Union[str, A.Compose]] = None,
        transform_config_val: Optional[Union[str, A.Compose]] = None,
        seed: Optional[int] = None,
        create_validation_set: bool = False,
    ) -> None:
        """Instantiate BTech Lightning Data Module.

        Args:
            root: Path to the BTech dataset
            category: Name of the BTech category.
            image_size: Variable to which image is resized.
            train_batch_size: Training batch size.
            test_batch_size: Testing batch size.
            num_workers: Number of workers.
            task: ``classification`` or ``segmentation``
            transform_config_train: Config for pre-processing during training.
            transform_config_val: Config for pre-processing during validation.
            seed: seed used for the random subset splitting
            create_validation_set: Create a validation subset in addition to the train and test subsets

        Examples
            >>> from anomalib.data import BTech
            >>> datamodule = BTech(
            ...     root="./datasets/BTech",
            ...     category="leather",
            ...     image_size=256,
            ...     train_batch_size=32,
            ...     test_batch_size=32,
            ...     num_workers=8,
            ...     transform_config_train=None,
            ...     transform_config_val=None,
            ... )
            >>> datamodule.setup()

            >>> i, data = next(enumerate(datamodule.train_dataloader()))
            >>> data.keys()
            dict_keys(['image'])
            >>> data["image"].shape
            torch.Size([32, 3, 256, 256])

            >>> i, data = next(enumerate(datamodule.val_dataloader()))
            >>> data.keys()
            dict_keys(['image_path', 'label', 'mask_path', 'image', 'mask'])
            >>> data["image"].shape, data["mask"].shape
            (torch.Size([32, 3, 256, 256]), torch.Size([32, 256, 256]))
        """
        super().__init__()

        self.root = root if isinstance(root, Path) else Path(root)
        self.category = category
        self.dataset_path = self.root / self.category
        self.transform_config_train = transform_config_train
        self.transform_config_val = transform_config_val
        self.image_size = image_size

        if self.transform_config_train is not None and self.transform_config_val is None:
            self.transform_config_val = self.transform_config_train

        self.pre_process_train = PreProcessor(config=self.transform_config_train, image_size=self.image_size)
        self.pre_process_val = PreProcessor(config=self.transform_config_val, image_size=self.image_size)

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        self.create_validation_set = create_validation_set
        self.task = task
        self.seed = seed

        self.train_data: Dataset
        self.test_data: Dataset
        if create_validation_set:
            self.val_data: Dataset
        self.inference_data: Dataset

    def prepare_data(self) -> None:
        """Download the dataset if not available."""
        if (self.root / self.category).is_dir():
            logger.info("Found the dataset.")
        else:
            zip_filename = self.root.parent / "btad.zip"

            logger.info("Downloading the BTech dataset.")
            with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="BTech") as progress_bar:
                urlretrieve(
                    url="https://avires.dimi.uniud.it/papers/btad/btad.zip",
                    filename=zip_filename,
                    reporthook=progress_bar.update_to,
                )  # nosec
            logger.info("Checking hash")
            hash_check(zip_filename, "c1fa4d56ac50dd50908ce04e81037a8e")
            logger.info("Extracting the dataset.")
            with zipfile.ZipFile(zip_filename, "r") as zip_file:
                zip_file.extractall(self.root.parent)

            logger.info("Renaming the dataset directory")
            shutil.move(src=str(self.root.parent / "BTech_Dataset_transformed"), dst=str(self.root))

            # NOTE: Each BTech category has different image extension as follows
            #       | Category | Image | Mask |
            #       |----------|-------|------|
            #       | 01       | bmp   | png  |
            #       | 02       | png   | png  |
            #       | 03       | bmp   | bmp  |
            # To avoid any conflict, the following script converts all the extensions to png.
            # This solution works fine, but it's also possible to properly ready the bmp and
            # png filenames from categories in `make_btech_dataset` function.
            logger.info("Convert the bmp formats to png to have consistent image extensions")
            for filename in tqdm(self.root.glob("**/*.bmp"), desc="Converting bmp to png"):
                image = cv2.imread(str(filename))
                cv2.imwrite(str(filename.with_suffix(".png")), image)
                filename.unlink()

            logger.info("Cleaning the tar file")
            zip_filename.unlink()

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup train, validation and test data.

        BTech dataset uses BTech dataset structure, which is the reason for
        using `anomalib.data.btech.BTech` class to get the dataset items.

        Args:
          stage: Optional[str]:  Train/Val/Test stages. (Default value = None)

        """
        logger.info("Setting up train, validation, test and prediction datasets.")
        if stage in (None, "fit"):
            self.train_data = BTechDataset(
                root=self.root,
                category=self.category,
                pre_process=self.pre_process_train,
                split="train",
                task=self.task,
                seed=self.seed,
                create_validation_set=self.create_validation_set,
            )

        if self.create_validation_set:
            self.val_data = BTechDataset(
                root=self.root,
                category=self.category,
                pre_process=self.pre_process_val,
                split="val",
                task=self.task,
                seed=self.seed,
                create_validation_set=self.create_validation_set,
            )

        self.test_data = BTechDataset(
            root=self.root,
            category=self.category,
            pre_process=self.pre_process_val,
            split="test",
            task=self.task,
            seed=self.seed,
            create_validation_set=self.create_validation_set,
        )

        if stage == "predict":
            self.inference_data = InferenceDataset(
                path=self.root, image_size=self.image_size, transform_config=self.transform_config_val
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
