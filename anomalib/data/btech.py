"""BTech Dataset.

This script contains PyTorch Lightning DataModule for the BTech dataset.

If the dataset is not on the file system, the script downloads and
extracts the dataset and create PyTorch data objects.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Union
from urllib.request import urlretrieve

import albumentations as A
import cv2
import pandas as pd
from pandas.core.frame import DataFrame
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from tqdm import tqdm

from anomalib.data.base import AnomalibDataModule
from anomalib.data.utils import DownloadProgressBar, hash_check
from anomalib.data.utils.split import (
    create_validation_set_from_test_set,
    split_normal_images_in_train_set,
)

logger = logging.getLogger(__name__)


@DATAMODULE_REGISTRY
class BTech(AnomalibDataModule):
    """BTechDataModule Lightning Data Module."""

    def __init__(
        self,
        root: str,
        category: str,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        train_batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 8,
        task: str = "segmentation",
        transform_config_train: Optional[Union[str, A.Compose]] = None,
        transform_config_val: Optional[Union[str, A.Compose]] = None,
        split_ratio: float = 0.2,
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
        self.root = root if isinstance(root, Path) else Path(root)
        self.category = category
        self.path = self.root / self.category

        self.create_validation_set = create_validation_set
        self.seed = seed
        self.split_ratio = split_ratio

        super().__init__(
            task=task,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            transform_config_train=transform_config_train,
            transform_config_val=transform_config_val,
            image_size=image_size,
            create_validation_set=create_validation_set,
        )

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

    def _create_samples(self) -> DataFrame:
        """Create BTech samples by parsing the BTech data file structure.

        The files are expected to follow the structure:
            path/to/dataset/category/split/[ok|ko]/image_filename.bmp
            path/to/dataset/category/ground_truth/ko/mask_filename.png

        This function creates a dataframe to store the parsed information based on the following format:
        |---|---------------|-------|---------|---------------|---------------------------------------|-------------|
        |   | path          | split | label   | image_path    | mask_path                             | label_index |
        |---|---------------|-------|---------|---------------|---------------------------------------|-------------|
        | 0 | datasets/name |  test |  ko     |  filename.png | ground_truth/ko/filename_mask.png     | 1           |
        |---|---------------|-------|---------|---------------|---------------------------------------|-------------|

        Returns:
            DataFrame: an output dataframe containing the samples of the dataset.
        """
        samples_list = [
            (str(self.path),) + filename.parts[-3:]
            for filename in self.path.glob("**/*")
            if filename.suffix in (".bmp", ".png")
        ]
        if len(samples_list) == 0:
            raise RuntimeError(f"Found 0 images in {self.path}")

        samples = pd.DataFrame(samples_list, columns=["path", "split", "label", "image_path"])
        samples = samples[samples.split != "ground_truth"]

        # Create mask_path column
        samples["mask_path"] = (
            samples.path
            + "/ground_truth/"
            + samples.label
            + "/"
            + samples.image_path.str.rstrip("bmp|png").str.rstrip(".")
            + ".png"
        )

        # Modify image_path column by converting to absolute path
        samples["image_path"] = samples.path + "/" + samples.split + "/" + samples.label + "/" + samples.image_path

        # Split the normal images in training set if test set doesn't
        # contain any normal images. This is needed because AUC score
        # cannot be computed based on 1-class
        if sum((samples.split == "test") & (samples.label == "ok")) == 0:
            samples = split_normal_images_in_train_set(samples, self.split_ratio, self.seed)

        # Good images don't have mask
        samples.loc[(samples.split == "test") & (samples.label == "ok"), "mask_path"] = ""

        # Create label index for normal (0) and anomalous (1) images.
        samples.loc[(samples.label == "ok"), "label_index"] = 0
        samples.loc[(samples.label != "ok"), "label_index"] = 1
        samples.label_index = samples.label_index.astype(int)

        if self.create_validation_set:
            samples = create_validation_set_from_test_set(samples, seed=self.seed)

        return samples
