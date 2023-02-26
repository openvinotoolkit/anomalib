"""MVTec 3D-AD Dataset (CC BY-NC-SA 4.0).

Description:
    This script contains PyTorch Dataset, Dataloader and PyTorch
        Lightning DataModule for the MVTec 3D-AD dataset.
    If the dataset is not on the file system, the script downloads and
        extracts the dataset and create PyTorch data objects.
License:
    MVTec 3D-AD dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0)(https://creativecommons.org/licenses/by-nc-sa/4.0/).
Reference:
    - Paul Bergmann, Xin Jin, David Sattlegger, Carsten Steger:
    The MVTec 3D-AD Dataset for Unsupervised 3D Anomaly Detection and Localization
    in: Proceedings of the 17th International Joint Conference on Computer Vision, Imaging
    and Computer Graphics Theory and Applications - Volume 5: VISAPP, 202-213, 2022,
    DOI: 10.5220/0010865000003124.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import albumentations as A
from pandas import DataFrame

from anomalib.data.base import AnomalibDataModule, AnomalibDepthDataset
from anomalib.data.task_type import TaskType
from anomalib.data.utils import (
    DownloadInfo,
    InputNormalizationMethod,
    Split,
    TestSplitMode,
    ValSplitMode,
    download_and_extract,
    get_transforms,
)

logger = logging.getLogger(__name__)


IMG_EXTENSIONS = [".png", ".PNG", ".tiff"]

DOWNLOAD_INFO = DownloadInfo(
    name="mvtec_3d",
    url="https://www.mydrive.ch/shares/45920/dd1eb345346df066c63b5c95676b961b/download/428824485-1643285832"
    "/mvtec_3d_anomaly_detection.tar.xz",
    hash="d8bb2800fbf3ac88e798da6ae10dc819",
)


def make_mvtec_3d_dataset(
    root: str | Path, split: str | Split | None = None, extensions: Sequence[str] | None = None
) -> DataFrame:
    """Create MVTec 3D-AD samples by parsing the MVTec AD data file structure.

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
        split (str | Split | None, optional): Dataset split (ie., either train or test). Defaults to None.
        split_ratio (float, optional): Ratio to split normal training images and add to the
            test set in case test set doesn't contain any normal images.
            Defaults to 0.1.
        seed (int, optional): Random seed to ensure reproducibility when splitting. Defaults to 0.
        create_validation_set (bool, optional): Boolean to create a validation set from the test set.
            MVTec AD dataset does not contain a validation set. Those wanting to create a validation set
            could set this flag to ``True``.

    Examples:
        The following example shows how to get training samples from MVTec 3D-AD bagel category:

        >>> root = Path('./MVTec3D')
        >>> category = 'bagel'
        >>> path = root / category
        >>> path
        PosixPath('MVTec3D/bagel')

        >>> samples = make_mvtec_3d_dataset(path, split='train', split_ratio=0.1, seed=0)
        >>> samples.head()
           path         split label image_path                           mask_path
        0  MVTec3D/bagel train good MVTec3D/bagel/train/good/rgb/105.png MVTec3D/bagel/ground_truth/good/gt/105.png
        1  MVTec3D/bagel train good MVTec3D/bagel/train/good/rgb/017.png MVTec3D/bagel/ground_truth/good/gt/017.png
        2  MVTec3D/bagel train good MVTec3D/bagel/train/good/rgb/137.png MVTec3D/bagel/ground_truth/good/gt/137.png
        3  MVTec3D/bagel train good MVTec3D/bagel/train/good/rgb/152.png MVTec3D/bagel/ground_truth/good/gt/152.png
        4  MVTec3D/bagel train good MVTec3D/bagel/train/good/rgb/109.png MVTec3D/bagel/ground_truth/good/gt/109.png
           depth_path                                   label_index
           MVTec3D/bagel/ground_truth/good/xyz/105.tiff 0
           MVTec3D/bagel/ground_truth/good/xyz/017.tiff 0
           MVTec3D/bagel/ground_truth/good/xyz/137.tiff 0
           MVTec3D/bagel/ground_truth/good/xyz/152.tiff 0
           MVTec3D/bagel/ground_truth/good/xyz/109.tiff 0

    Returns:
        DataFrame: an output dataframe containing the samples of the dataset.
    """
    if extensions is None:
        extensions = IMG_EXTENSIONS

    root = Path(root)
    samples_list = [(str(root),) + f.parts[-4:] for f in root.glob(r"**/*") if f.suffix in extensions]
    if not samples_list:
        raise RuntimeError(f"Found 0 images in {root}")

    samples = DataFrame(samples_list, columns=["path", "split", "label", "type", "file_name"])

    # Modify image_path column by converting to absolute path
    samples.loc[(samples.type == "rgb"), "image_path"] = (
        samples.path + "/" + samples.split + "/" + samples.label + "/" + "rgb/" + samples.file_name
    )
    samples.loc[(samples.type == "rgb"), "depth_path"] = (
        samples.path
        + "/"
        + samples.split
        + "/"
        + samples.label
        + "/"
        + "xyz/"
        + samples.file_name.str.split(".").str[0]
        + ".tiff"
    )

    # Create label index for normal (0) and anomalous (1) images.
    samples.loc[(samples.label == "good"), "label_index"] = 0
    samples.loc[(samples.label != "good"), "label_index"] = 1
    samples.label_index = samples.label_index.astype(int)

    # separate masks from samples
    mask_samples = samples.loc[((samples.split == "test") & (samples.type == "rgb"))].sort_values(
        by="image_path", ignore_index=True
    )
    samples = samples.sort_values(by="image_path", ignore_index=True)

    # assign mask paths to all test images
    samples.loc[((samples.split == "test") & (samples.type == "rgb")), "mask_path"] = (
        mask_samples.path + "/" + samples.split + "/" + samples.label + "/" + "gt/" + samples.file_name
    )
    samples.dropna(subset=["image_path"], inplace=True)
    samples = samples.astype({"image_path": "str", "mask_path": "str", "depth_path": "str"})

    # assert that the right mask files are associated with the right test images
    assert (
        samples.loc[samples.label_index == 1]
        .apply(lambda x: Path(x.image_path).stem in Path(x.mask_path).stem, axis=1)
        .all()
    ), "Mismatch between anomalous images and ground truth masks. Make sure the mask files in 'ground_truth' \
              folder follow the same naming convention as the anomalous images in the dataset (e.g. image: '000.png', \
              mask: '000.png' or '000_mask.png')."

    # assert that the right depth image files are associated with the right test images
    assert (
        samples.loc[samples.label_index == 1]
        .apply(lambda x: Path(x.image_path).stem in Path(x.depth_path).stem, axis=1)
        .all()
    ), "Mismatch between anomalous images and depth images. Make sure the mask files in 'xyz' \
              folder follow the same naming convention as the anomalous images in the dataset (e.g. image: '000.png', \
              depth: '000.tiff')."

    if split:
        samples = samples[samples.split == split].reset_index(drop=True)

    return samples


class MVTec3DDataset(AnomalibDepthDataset):
    """MVTec 3D dataset class.

    Args:
        task (TaskType): Task type, ``classification``, ``detection`` or ``segmentation``
        transform (A.Compose): Albumentations Compose object describing the transforms that are applied to the inputs.
        split (str | Split | None): Split of the dataset, usually Split.TRAIN or Split.TEST
        root (Path | str): Path to the root of the dataset
        category (str): Sub-category of the dataset, e.g. 'bagel'
    """

    def __init__(
        self,
        task: TaskType,
        transform: A.Compose,
        root: Path | str,
        category: str,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(task=task, transform=transform)

        self.root_category = Path(root) / Path(category)
        self.split = split

    def _setup(self) -> None:
        self.samples = make_mvtec_3d_dataset(self.root_category, split=self.split, extensions=IMG_EXTENSIONS)


class MVTec3D(AnomalibDataModule):
    """MVTec Datamodule.

    Args:
        root (Path | str): Path to the root of the dataset
        category (str): Category of the MVTec dataset (e.g. "bottle" or "cable").
        image_size (int | tuple[int, int] | None, optional): Size of the input image.
            Defaults to None.
        center_crop (int | tuple[int, int] | None, optional): When provided, the images will be center-cropped
            to the provided dimensions.
        normalize (bool): When True, the images will be normalized to the ImageNet statistics.
        train_batch_size (int, optional): Training batch size. Defaults to 32.
        eval_batch_size (int, optional): Test batch size. Defaults to 32.
        num_workers (int, optional): Number of workers. Defaults to 8.
        task TaskType): Task type, 'classification', 'detection' or 'segmentation'
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
        seed (int | None, optional): Seed which may be set to a fixed value for reproducibility.
    """

    def __init__(
        self,
        root: Path | str,
        category: str,
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
        val_split_mode: ValSplitMode = ValSplitMode.SAME_AS_TEST,
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

        self.root = Path(root)
        self.category = Path(category)

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

        self.train_data = MVTec3DDataset(
            task=task, transform=transform_train, split=Split.TRAIN, root=root, category=category
        )
        self.test_data = MVTec3DDataset(
            task=task, transform=transform_eval, split=Split.TEST, root=root, category=category
        )

    def prepare_data(self) -> None:
        """Download the dataset if not available."""
        if (self.root / self.category).is_dir():
            logger.info("Found the dataset.")
        else:
            download_and_extract(self.root, DOWNLOAD_INFO)
