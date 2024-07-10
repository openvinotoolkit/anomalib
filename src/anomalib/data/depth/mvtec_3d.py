"""MVTec 3D-AD Dataset (CC BY-NC-SA 4.0).

Description:
    This script contains PyTorch Dataset, Dataloader and PyTorch Lightning DataModule for the MVTec 3D-AD dataset.
    If the dataset is not on the file system, the script downloads and extracts the dataset and create PyTorch data
    objects.

License:
    MVTec 3D-AD dataset is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
        License (CC BY-NC-SA 4.0)(https://creativecommons.org/licenses/by-nc-sa/4.0/).

Reference:
    - Paul Bergmann, Xin Jin, David Sattlegger, Carsten Steger: The MVTec 3D-AD Dataset for Unsupervised 3D Anomaly
        Detection and Localization in: Proceedings of the 17th International Joint Conference on Computer Vision,
        Imaging and Computer Graphics Theory and Applications - Volume 5: VISAPP, 202-213, 2022, DOI: 10.5220/
        0010865000003124.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence
from pathlib import Path

from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib import TaskType
from anomalib.data.base import AnomalibDataModule, AnomalibDepthDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import (
    DownloadInfo,
    LabelName,
    Split,
    TestSplitMode,
    ValSplitMode,
    download_and_extract,
    validate_path,
)

logger = logging.getLogger(__name__)


IMG_EXTENSIONS = [".png", ".PNG", ".tiff"]

DOWNLOAD_INFO = DownloadInfo(
    name="mvtec_3d",
    url="https://www.mydrive.ch/shares/45920/dd1eb345346df066c63b5c95676b961b/download/428824485-1643285832"
    "/mvtec_3d_anomaly_detection.tar.xz",
    hashsum="d8bb2800fbf3ac88e798da6ae10dc819",
)

CATEGORIES = ("bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire")


def make_mvtec_3d_dataset(
    root: str | Path,
    split: str | Split | None = None,
    extensions: Sequence[str] | None = None,
) -> DataFrame:
    """Create MVTec 3D-AD samples by parsing the MVTec AD data file structure.

    The files are expected to follow this structure:
    - `path/to/dataset/split/category/image_filename.png`
    - `path/to/dataset/ground_truth/category/mask_filename.png`

    This function creates a DataFrame to store the parsed information. The DataFrame follows this format:

    +---+---------------+-------+---------+---------------+---------------------------------------+-------------+
    |   | path          | split | label   | image_path    | mask_path                             | label_index |
    +---+---------------+-------+---------+---------------+---------------------------------------+-------------+
    | 0 | datasets/name | test  | defect  | filename.png  | ground_truth/defect/filename_mask.png | 1           |
    +---+---------------+-------+---------+---------------+---------------------------------------+-------------+

    Args:
        root (Path): Path to the dataset.
        split (str | Split | None, optional): Dataset split (e.g., 'train' or 'test').
            Defaults to ``None``.
        extensions (Sequence[str] | None, optional): List of file extensions to be included in the dataset.
            Defaults to ``None``.

    Examples:
        The following example shows how to get training samples from the MVTec 3D-AD 'bagel' category:

        >>> from pathlib import Path
        >>> root = Path('./MVTec3D')
        >>> category = 'bagel'
        >>> path = root / category
        >>> print(path)
        PosixPath('MVTec3D/bagel')

        >>> samples = create_mvtec_3d_ad_samples(path, split='train')
        >>> print(samples.head())
            path          split label image_path                          mask_path                        label_index
            MVTec3D/bagel train good MVTec3D/bagel/train/good/rgb/105.png MVTec3D/bagel/ground_truth/good/gt/105.png 0
            MVTec3D/bagel train good MVTec3D/bagel/train/good/rgb/017.png MVTec3D/bagel/ground_truth/good/gt/017.png 0

    Returns:
        DataFrame: An output DataFrame containing the samples of the dataset.
    """
    if extensions is None:
        extensions = IMG_EXTENSIONS

    root = validate_path(root)
    samples_list = [(str(root),) + f.parts[-4:] for f in root.glob(r"**/*") if f.suffix in extensions]
    if not samples_list:
        msg = f"Found 0 images in {root}"
        raise RuntimeError(msg)

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
    samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
    samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype(int)

    # separate masks from samples
    mask_samples = samples.loc[((samples.split == "test") & (samples.type == "rgb"))].sort_values(
        by="image_path",
        ignore_index=True,
    )
    samples = samples.sort_values(by="image_path", ignore_index=True)

    # assign mask paths to all test images
    samples.loc[((samples.split == "test") & (samples.type == "rgb")), "mask_path"] = (
        mask_samples.path + "/" + samples.split + "/" + samples.label + "/" + "gt/" + samples.file_name
    )
    samples = samples.dropna(subset=["image_path"])
    samples = samples.astype({"image_path": "str", "mask_path": "str", "depth_path": "str"})

    # assert that the right mask files are associated with the right test images
    mismatch_masks = (
        samples.loc[samples.label_index == LabelName.ABNORMAL]
        .apply(lambda x: Path(x.image_path).stem in Path(x.mask_path).stem, axis=1)
        .all()
    )
    if not mismatch_masks:
        msg = """Mismatch between anomalous images and ground truth masks. Make sure the mask files
          in 'ground_truth' folder follow the same naming convention as the anomalous images in
          the dataset (e.g. image: '000.png', mask: '000.png' or '000_mask.png')."""
        raise MisMatchError(msg)

    mismatch_depth = (
        samples.loc[samples.label_index == LabelName.ABNORMAL]
        .apply(lambda x: Path(x.image_path).stem in Path(x.depth_path).stem, axis=1)
        .all()
    )
    if not mismatch_depth:
        msg = """Mismatch between anomalous images and depth images. Make sure the mask files in
          'xyz' folder follow the same naming convention as the anomalous images in the dataset
          (e.g. image: '000.png', depth: '000.tiff')."""
        raise MisMatchError(msg)

    if split:
        samples = samples[samples.split == split].reset_index(drop=True)

    return samples


class MVTec3DDataset(AnomalibDepthDataset):
    """MVTec 3D dataset class.

    Args:
        task (TaskType): Task type, ``classification``, ``detection`` or ``segmentation``
        root (Path | str): Path to the root of the dataset
            Defaults to ``"./datasets/MVTec3D"``.
        category (str): Sub-category of the dataset, e.g. 'bagel'
            Defaults to ``"bagel"``.
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
        split (str | Split | None): Split of the dataset, usually Split.TRAIN or Split.TEST
            Defaults to ``None``.
    """

    def __init__(
        self,
        task: TaskType,
        root: Path | str = "./datasets/MVTec3D",
        category: str = "bagel",
        transform: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(task=task, transform=transform)

        self.root_category = Path(root) / Path(category)
        self.split = split
        self.samples = make_mvtec_3d_dataset(self.root_category, split=self.split, extensions=IMG_EXTENSIONS)


class MVTec3D(AnomalibDataModule):
    """MVTec Datamodule.

    Args:
        root (Path | str): Path to the root of the dataset
            Defaults to ``"./datasets/MVTec3D"``.
        category (str): Category of the MVTec dataset (e.g. "bottle" or "cable").
            Defaults to ``bagel``.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Test batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            Defaults to ``8``.
        task (TaskType): Task type, 'classification', 'detection' or 'segmentation'
            Defaults to ``TaskType.SEGMENTATION``.
        image_size (tuple[int, int], optional): Size to which input images should be resized.
            Defaults to ``None``.
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
        train_transform (Transform, optional): Transforms that should be applied to the input images during training.
            Defaults to ``None``.
        eval_transform (Transform, optional): Transforms that should be applied to the input images during evaluation.
            Defaults to ``None``.
        test_split_mode (TestSplitMode): Setting that determines how the testing subset is obtained.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float): Fraction of images from the train set that will be reserved for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
            Defaults to ``ValSplitMode.SAME_AS_TEST``.
        val_split_ratio (float): Fraction of train or test images that will be reserved for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Seed which may be set to a fixed value for reproducibility.
            Defaults to ``None``.
    """

    def __init__(
        self,
        root: Path | str = "./datasets/MVTec3D",
        category: str = "bagel",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        task: TaskType | str = TaskType.SEGMENTATION,
        image_size: tuple[int, int] | None = None,
        transform: Transform | None = None,
        train_transform: Transform | None = None,
        eval_transform: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode | str = ValSplitMode.SAME_AS_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            image_size=image_size,
            transform=transform,
            train_transform=train_transform,
            eval_transform=eval_transform,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        self.task = TaskType(task)
        self.root = Path(root)
        self.category = category

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = MVTec3DDataset(
            task=self.task,
            transform=self.train_transform,
            split=Split.TRAIN,
            root=self.root,
            category=self.category,
        )
        self.test_data = MVTec3DDataset(
            task=self.task,
            transform=self.eval_transform,
            split=Split.TEST,
            root=self.root,
            category=self.category,
        )

    def prepare_data(self) -> None:
        """Download the dataset if not available."""
        if (self.root / self.category).is_dir():
            logger.info("Found the dataset.")
        else:
            download_and_extract(self.root, DOWNLOAD_INFO)
