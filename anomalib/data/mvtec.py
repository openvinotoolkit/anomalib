"""MVTec AD Dataset (CC BY-NC-SA 4.0).

Description:
    This script contains PyTorch Dataset, Dataloader and PyTorch
        Lightning DataModule for the MVTec AD dataset.
    If the dataset is not on the file system, the script downloads and
        extracts the dataset and create PyTorch data objects.
License:
    MVTec AD dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0)(https://creativecommons.org/licenses/by-nc-sa/4.0/).
Reference:
    - Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger:
      The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for
      Unsupervised Anomaly Detection; in: International Journal of Computer Vision
      129(4):1038-1059, 2021, DOI: 10.1007/s11263-020-01400-4.
    - Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger: MVTec AD —
      A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection;
      in: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
      9584-9592, 2019, DOI: 10.1109/CVPR.2019.00982.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import tarfile
from pathlib import Path
from typing import Optional, Tuple, Union
from urllib.request import urlretrieve

import albumentations as A
from pandas import DataFrame

from anomalib.data.base import AnomalibDataModule, AnomalibDataset
from anomalib.data.task_type import TaskType
from anomalib.data.utils import (
    DownloadProgressBar,
    Split,
    TestSplitMode,
    ValSplitMode,
    hash_check,
)
from anomalib.pre_processing import PreProcessor

logger = logging.getLogger(__name__)


def make_mvtec_dataset(root: Union[str, Path], split: Optional[Union[Split, str]] = None) -> DataFrame:
    """Create MVTec AD samples by parsing the MVTec AD data file structure.

    The files are expected to follow the structure:
        path/to/dataset/split/category/image_filename.png
        path/to/dataset/ground_truth/category/mask_filename.png

    This function creates a dataframe to store the parsed information based on the following format:
    |---|---------------|-------|---------|---------------|---------------------------------------|-------------|
    |   | path          | split | label   | image_path    | mask_path                             | label_index |
    |---|---------------|-------|---------|---------------|---------------------------------------|-------------|
    | 0 | datasets/name |  test |  defect |  filename.png | ground_truth/defect/filename_mask.png | 1           |
    |---|---------------|-------|---------|---------------|---------------------------------------|-------------|

    Args:
        path (Path): Path to dataset
        split (Optional[Union[Split, str]], optional): Dataset split (ie., either train or test). Defaults to None.
        split_ratio (float, optional): Ratio to split normal training images and add to the
            test set in case test set doesn't contain any normal images.
            Defaults to 0.1.
        seed (int, optional): Random seed to ensure reproducibility when splitting. Defaults to 0.
        create_validation_set (bool, optional): Boolean to create a validation set from the test set.
            MVTec AD dataset does not contain a validation set. Those wanting to create a validation set
            could set this flag to ``True``.

    Examples:
        The following example shows how to get training samples from MVTec AD bottle category:

        >>> root = Path('./MVTec')
        >>> category = 'bottle'
        >>> path = root / category
        >>> path
        PosixPath('MVTec/bottle')

        >>> samples = make_mvtec_dataset(path, split='train', split_ratio=0.1, seed=0)
        >>> samples.head()
           path         split label image_path                           mask_path                   label_index
        0  MVTec/bottle train good MVTec/bottle/train/good/105.png MVTec/bottle/ground_truth/good/105_mask.png 0
        1  MVTec/bottle train good MVTec/bottle/train/good/017.png MVTec/bottle/ground_truth/good/017_mask.png 0
        2  MVTec/bottle train good MVTec/bottle/train/good/137.png MVTec/bottle/ground_truth/good/137_mask.png 0
        3  MVTec/bottle train good MVTec/bottle/train/good/152.png MVTec/bottle/ground_truth/good/152_mask.png 0
        4  MVTec/bottle train good MVTec/bottle/train/good/109.png MVTec/bottle/ground_truth/good/109_mask.png 0

    Returns:
        DataFrame: an output dataframe containing the samples of the dataset.
    """
    samples_list = [(str(root),) + filename.parts[-3:] for filename in Path(root).glob("**/*.png")]
    if len(samples_list) == 0:
        raise RuntimeError(f"Found 0 images in {root}")

    samples = DataFrame(samples_list, columns=["path", "split", "label", "image_path"])
    samples = samples[samples.split != "ground_truth"]

    # Create mask_path column
    samples["mask_path"] = (
        samples.path
        + "/ground_truth/"
        + samples.label
        + "/"
        + samples.image_path.str.rstrip("png").str.rstrip(".")
        + "_mask.png"
    )

    # Modify image_path column by converting to absolute path
    samples["image_path"] = samples.path + "/" + samples.split + "/" + samples.label + "/" + samples.image_path

    # Good images don't have mask
    samples.loc[(samples.split == "test") & (samples.label == "good"), "mask_path"] = ""

    # Create label index for normal (0) and anomalous (1) images.
    samples.loc[(samples.label == "good"), "label_index"] = 0
    samples.loc[(samples.label != "good"), "label_index"] = 1
    samples.label_index = samples.label_index.astype(int)

    if split:
        samples = samples[samples.split == split].reset_index(drop=True)

    return samples


class MVTecDataset(AnomalibDataset):
    """MVTec dataset class.

    Args:
        task (TaskType): Task type, ``classification``, ``detection`` or ``segmentation``
        pre_process (PreProcessor): Pre-processor object
        split (Optional[Union[Split, str]]): Split of the dataset, usually Split.TRAIN or Split.TEST
        root (str): Path to the root of the dataset
        category (str): Sub-category of the dataset, e.g. 'bottle'
    """

    def __init__(
        self,
        task: TaskType,
        pre_process: PreProcessor,
        root: str,
        category: str,
        split: Optional[Union[Split, str]] = None,
    ) -> None:
        super().__init__(task=task, pre_process=pre_process)

        self.root_category = Path(root) / Path(category)
        self.split = split

    def _setup(self):
        self.samples = make_mvtec_dataset(self.root_category, split=self.split)


class MVTec(AnomalibDataModule):
    """MVTec Datamodule.

    Args:
        root (str): Path to the root of the dataset
        category (str): Category of the MVTec dataset (e.g. "bottle" or "cable").
        image_size (Optional[Union[int, Tuple[int, int]]], optional): Size of the input image.
            Defaults to None.
        train_batch_size (int, optional): Training batch size. Defaults to 32.
        eval_batch_size (int, optional): Test batch size. Defaults to 32.
        num_workers (int, optional): Number of workers. Defaults to 8.
        task TaskType): Task type, 'classification', 'detection' or 'segmentation'
        transform_config_train (Optional[Union[str, A.Compose]], optional): Config for pre-processing
            during training.
            Defaults to None.
        transform_config_val (Optional[Union[str, A.Compose]], optional): Config for pre-processing
            during validation.
            Defaults to None.
        test_split_mode (TestSplitMode): Setting that determines how the testing subset is obtained.
        test_split_ratio (float): Fraction of images from the train set that will be reserved for testing.
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
        val_split_ratio (float): Fraction of train or test images that will be reserved for validation.
        seed (Optional[int], optional): Seed which may be set to a fixed value for reproducibility.
    """

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
        test_split_mode: TestSplitMode = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode = ValSplitMode.SAME_AS_TEST,
        val_split_ratio: float = 0.5,
        seed: Optional[int] = None,
    ):
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

        self.root = Path(root)
        self.category = Path(category)

        # TODO: Get rid of PreProcessor by passing transform directly
        pre_process_train = PreProcessor(config=transform_config_train, image_size=image_size)
        pre_process_eval = PreProcessor(config=transform_config_eval, image_size=image_size)

        self.train_data = MVTecDataset(
            task=task, pre_process=pre_process_train, split=Split.TRAIN, root=root, category=category
        )
        self.test_data = MVTecDataset(
            task=task, pre_process=pre_process_eval, split=Split.TEST, root=root, category=category
        )

    def prepare_data(self) -> None:
        """Download the dataset if not available."""
        if (self.root / self.category).is_dir():
            logger.info("Found the dataset.")
        else:
            self.root.mkdir(parents=True, exist_ok=True)

            logger.info("Downloading the Mvtec AD dataset.")
            url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094"
            dataset_name = "mvtec_anomaly_detection.tar.xz"
            zip_filename = self.root / dataset_name
            with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="MVTec AD") as progress_bar:
                urlretrieve(
                    url=f"{url}/{dataset_name}",
                    filename=zip_filename,
                    reporthook=progress_bar.update_to,
                )
            logger.info("Checking hash")
            hash_check(zip_filename, "eefca59f2cede9c3fc5b6befbfec275e")

            logger.info("Extracting the dataset.")
            with tarfile.open(zip_filename) as tar_file:
                tar_file.extractall(self.root)

            logger.info("Cleaning the tar file")
            (zip_filename).unlink()
