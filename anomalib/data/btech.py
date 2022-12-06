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

from anomalib.data.base import AnomalibDataModule, AnomalibDataset
from anomalib.data.task_type import TaskType
from anomalib.data.utils import DownloadProgressBar, Split, ValSplitMode, hash_check
from anomalib.pre_processing import PreProcessor

logger = logging.getLogger(__name__)


def make_btech_dataset(path: Path, split: Optional[Union[Split, str]] = None) -> DataFrame:
    """Create BTech samples by parsing the BTech data file structure.

    The files are expected to follow the structure:
        path/to/dataset/split/category/image_filename.png
        path/to/dataset/ground_truth/category/mask_filename.png

    Args:
        path (Path): Path to dataset
        split (Optional[Union[Split, str]], optional): Dataset split (ie., either train or test). Defaults to None.
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

    # Good images don't have mask
    samples.loc[(samples.split == "test") & (samples.label == "ok"), "mask_path"] = ""

    # Create label index for normal (0) and anomalous (1) images.
    samples.loc[(samples.label == "ok"), "label_index"] = 0
    samples.loc[(samples.label != "ok"), "label_index"] = 1
    samples.label_index = samples.label_index.astype(int)

    # Get the data frame for the split.
    if split:
        samples = samples[samples.split == split]
        samples = samples.reset_index(drop=True)

    return samples


class BTechDataset(AnomalibDataset):
    """BTech PyTorch Dataset."""

    def __init__(
        self,
        root: Union[Path, str],
        category: str,
        pre_process: PreProcessor,
        split: Optional[Union[Split, str]] = None,
        task: TaskType = TaskType.SEGMENTATION,
    ) -> None:
        """Btech Dataset class.

        Args:
            root: Path to the BTech dataset
            category: Name of the BTech category.
            pre_process: List of pre_processing object containing albumentation compose.
            split: 'train', 'val' or 'test'
            task: ``classification``, ``detection`` or ``segmentation``
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
        super().__init__(task, pre_process)

        self.root_category = Path(root) / category
        self.split = split

    def _setup(self):
        self.samples = make_btech_dataset(path=self.root_category, split=self.split)


@DATAMODULE_REGISTRY
class BTech(AnomalibDataModule):
    """BTechDataModule Lightning Data Module."""

    def __init__(
        self,
        root: str,
        category: str,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        task: TaskType = TaskType.SEGMENTATION,
        transform_config_train: Optional[Union[str, A.Compose]] = None,
        transform_config_eval: Optional[Union[str, A.Compose]] = None,
        val_split_mode: ValSplitMode = ValSplitMode.SAME_AS_TEST,
        val_split_ratio: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        """Instantiate BTech Lightning Data Module.

        Args:
            root: Path to the BTech dataset
            category: Name of the BTech category.
            image_size: Variable to which image is resized.
            train_batch_size: Training batch size.
            test_batch_size: Testing batch size.
            num_workers: Number of workers.
            task: ``classification``, ``detection`` or ``segmentation``
            transform_config_train: Config for pre-processing during training.
            transform_config_val: Config for pre-processing during validation.
            create_validation_set: Create a validation subset in addition to the train and test subsets
            seed (Optional[int], optional): Seed used during random subset splitting.

        Examples:
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
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        self.root = Path(root)
        self.category = Path(category)

        pre_process_train = PreProcessor(config=transform_config_train, image_size=image_size)
        pre_process_eval = PreProcessor(config=transform_config_eval, image_size=image_size)

        self.train_data = BTechDataset(
            task=task, pre_process=pre_process_train, split=Split.TRAIN, root=root, category=category
        )
        self.test_data = BTechDataset(
            task=task, pre_process=pre_process_eval, split=Split.TEST, root=root, category=category
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
