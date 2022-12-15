"""CUHK Avenue Dataset.

Description:
    This module contains PyTorch Dataset and PyTorch
        Lightning DataModule for the CUHK Avenue dataset.
    If the dataset is not on the file system, the DataModule class downloads and
        extracts the dataset and converts the .mat mask files to .png format.
Reference:
    - Lu, Cewu, Jianping Shi, and Jiaya Jia. "Abnormal event detection at 150 fps in matlab."
    Proceedings of the IEEE international conference on computer vision. 2013.
"""

import logging
import math
import zipfile
from collections import namedtuple
from pathlib import Path
from shutil import move
from typing import Callable, Optional, Tuple, Union
from urllib.request import urlretrieve

import albumentations as A
import cv2
import numpy as np
import scipy.io
from pandas import DataFrame
from torch import Tensor

from anomalib.data.base import AnomalibVideoDataModule, AnomalibVideoDataset
from anomalib.data.task_type import TaskType
from anomalib.data.utils import (
    DownloadProgressBar,
    Split,
    ValSplitMode,
    get_transforms,
    hash_check,
)
from anomalib.data.utils.video import ClipsIndexer

logger = logging.getLogger(__name__)


# define some info used for downloading the dataset
DownloadInfo = namedtuple("DownloadInfo", "description zip_filename expected_hash extracted_folder_name")
DATASET_DOWNLOAD_INFO = DownloadInfo(
    "dataset", "Avenue_Dataset.zip", "b7a34b212ecdd30efbd989a6dcb1aceb", "Avenue Dataset"
)
ANNOTATIONS_DOWNLOAD_INFO = DownloadInfo(
    "annotations", "ground_truth_demo.zip", "e8e3bff99195b6b511534083b9dbe1f5", "ground_truth_demo"
)


def make_avenue_dataset(root: Path, gt_dir: Path, split: Optional[Union[Split, str]] = None) -> DataFrame:
    """Create CUHK Avenue dataset by parsing the file structure.

    The files are expected to follow the structure:
        path/to/dataset/[training_videos|testing_videos]/video_filename.avi
        path/to/ground_truth/mask_filename.mat

    Args:
        root (Path): Path to dataset
        gt_dir (Path): Path to the ground truth
        split (Optional[Union[Split, str]], optional): Dataset split (ie., either train or test). Defaults to None.

    Example:
        The following example shows how to get testing samples from Avenue dataset:

        >>> root = Path('./avenue')
        >>> gt_dir = Path('./avenue/masks')
        >>> samples = make_avenue_dataset(path, gt_dir, split='test')
        >>> samples.head()
           root     folder         image_path                      mask_path                   split
        0  ./avenue testing_videos ./avenue/training_videos/01.avi ./avenue/masks/01_label.mat test
        1  ./avenue testing_videos ./avenue/training_videos/02.avi ./avenue/masks/01_label.mat test
        ...

    Returns:
        DataFrame: an output dataframe containing samples for the requested split (ie., train or test)
    """

    samples_list = [(str(root),) + filename.parts[-2:] for filename in Path(root).glob("**/*.avi")]
    samples = DataFrame(samples_list, columns=["root", "folder", "image_path"])

    samples.loc[samples.folder == "testing_videos", "mask_path"] = (
        samples.image_path.str.split(".").str[0].str.lstrip("0") + "_label.mat"
    )
    samples.loc[samples.folder == "testing_videos", "mask_path"] = (
        str(gt_dir) + "/testing_label_mask/" + samples.mask_path
    )
    samples.loc[samples.folder == "training_videos", "mask_path"] = ""

    samples["image_path"] = samples.root + "/" + samples.folder + "/" + samples.image_path

    samples.loc[samples.folder == "training_videos", "split"] = "train"
    samples.loc[samples.folder == "testing_videos", "split"] = "test"

    if split:
        samples = samples[samples.split == split]
        samples = samples.reset_index(drop=True)

    return samples


class AvenueClipsIndexer(ClipsIndexer):
    """Clips class for UCSDped dataset."""

    def get_mask(self, idx) -> Optional[Tensor]:
        """Retrieve the masks from the file system."""

        video_idx, frames_idx = self.get_clip_location(idx)
        matfile = self.mask_paths[video_idx]
        if matfile == "":  # no gt masks available for this clip
            return None
        frames = self.clips[video_idx][frames_idx]

        # read masks from .png files if available, othwerise from mat files.
        mask_folder = Path(matfile).with_suffix("")
        if mask_folder.exists():
            mask_frames = sorted(mask_folder.glob("*"))
            mask_paths = [mask_frames[idx] for idx in frames.int()]
            masks = np.stack([cv2.imread(str(mask_path), flags=0) for mask_path in mask_paths])
        else:
            mat = scipy.io.loadmat(matfile)
            masks = np.vstack([np.stack(m) for m in mat["volLabel"]])
            masks = np.take(masks, frames, 0)
        return masks


class AvenueDataset(AnomalibVideoDataset):
    """Avenue Dataset class.

    Args:
        task (TaskType): Task type, 'classification', 'detection' or 'segmentation'
        root (str): Path to the root of the dataset
        gt_dir (str): Path to the ground truth files
        transform (A.Compose): Albumentations Compose object describing the transforms that are applied to the inputs.
        split (Optional[Union[Split, str]]): Split of the dataset, usually Split.TRAIN or Split.TEST
        clip_length_in_frames (int, optional): Number of video frames in each clip.
        frames_between_clips (int, optional): Number of frames between each consecutive video clip.
    """

    def __init__(
        self,
        task: TaskType,
        root: Union[Path, str],
        gt_dir: str,
        transform: A.Compose,
        split: Split,
        clip_length_in_frames: int = 1,
        frames_between_clips: int = 1,
    ):
        super().__init__(task, transform, clip_length_in_frames, frames_between_clips)

        self.root = root
        self.gt_dir = gt_dir
        self.split = split
        self.indexer_cls: Callable = AvenueClipsIndexer

    def _setup(self):
        """Create and assign samples."""
        self.samples = make_avenue_dataset(self.root, self.gt_dir, self.split)


class Avenue(AnomalibVideoDataModule):
    """Avenue DataModule class.

    Args:
        root (str): Path to the root of the dataset
        gt_dir (str): Path to the ground truth files
        clip_length_in_frames (int, optional): Number of video frames in each clip.
        frames_between_clips (int, optional): Number of frames between each consecutive video clip.
        task TaskType): Task type, 'classification', 'detection' or 'segmentation'
        image_size (Optional[Union[int, Tuple[int, int]]], optional): Size of the input image.
            Defaults to None.
        center_crop (Optional[Union[int, Tuple[int, int]]], optional): When provided, the images will be center-cropped
            to the provided dimensions.
        normalize (bool): When True, the images will be normalized to the ImageNet statistics.
        train_batch_size (int, optional): Training batch size. Defaults to 32.
        eval_batch_size (int, optional): Test batch size. Defaults to 32.
        num_workers (int, optional): Number of workers. Defaults to 8.
        transform_config_train (Optional[Union[str, A.Compose]], optional): Config for pre-processing
            during training.
            Defaults to None.
        transform_config_val (Optional[Union[str, A.Compose]], optional): Config for pre-processing
            during validation.
            Defaults to None.
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
        val_split_ratio (float): Fraction of train or test images that will be reserved for validation.
        seed (Optional[int], optional): Seed which may be set to a fixed value for reproducibility.
    """

    def __init__(
        self,
        root: str,
        gt_dir: str,
        clip_length_in_frames: int = 1,
        frames_between_clips: int = 1,
        task: TaskType = TaskType.SEGMENTATION,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        center_crop: Optional[Union[int, Tuple[int, int]]] = None,
        normalize: bool = True,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        transform_config_train: Optional[Union[str, A.Compose]] = None,
        transform_config_eval: Optional[Union[str, A.Compose]] = None,
        val_split_mode: ValSplitMode = ValSplitMode.FROM_TEST,
        val_split_ratio: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        self.root = Path(root)
        self.gt_dir = Path(gt_dir)

        transform_train = get_transforms(
            config=transform_config_train, image_size=image_size, center_crop=center_crop, normalize=normalize
        )
        transform_eval = get_transforms(
            config=transform_config_eval, image_size=image_size, center_crop=center_crop, normalize=normalize
        )

        self.train_data = AvenueDataset(
            task=task,
            transform=transform_train,
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            root=root,
            gt_dir=gt_dir,
            split=Split.TRAIN,
        )

        self.test_data = AvenueDataset(
            task=task,
            transform=transform_eval,
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            root=root,
            gt_dir=gt_dir,
            split=Split.TEST,
        )

    def prepare_data(self) -> None:
        """Download the dataset and ground truth if not available, and convert mask files to a more usable format."""
        # download dataset
        self._download_and_extract(self.root, DATASET_DOWNLOAD_INFO)
        # download annotations
        self._download_and_extract(self.gt_dir, ANNOTATIONS_DOWNLOAD_INFO)
        # convert masks
        self._convert_masks(self.gt_dir)

    @staticmethod
    def _convert_masks(gt_dir: Path):
        """Convert mask files to .png.

        The masks in the Avenue datasets are provided as matlab (.mat) files. To speed up data loading, we convert the
        masks into a sepaarte .png file for every video frame in the dataset.

        Args:
            gt_dir (Path): Ground truth folder of the dataset.
        """
        # convert masks to numpy
        masks_dir = gt_dir / "testing_label_mask"
        # get file names
        mat_files = list(masks_dir.glob("*.mat"))
        mask_folders = [matfile.with_suffix("") for matfile in mat_files]
        if not all(folder.exists() for folder in mask_folders):
            # convert mask files to images
            logger.info("converting mat files to .png format.")
            for mat_file, mask_folder in zip(mat_files, mask_folders):
                mat = scipy.io.loadmat(mat_file)
                mask_folder.mkdir(parents=True, exist_ok=True)
                masks = mat["volLabel"].squeeze()
                for idx, mask in enumerate(masks):
                    filename = (mask_folder / str(idx).zfill(int(math.log10(len(masks)) + 1))).with_suffix(".png")
                    cv2.imwrite(str(filename), mask)

    @staticmethod
    def _download_and_extract(root: Path, info: DownloadInfo):
        """Download and extract the dataset or the annotations.

        Args:
            root (Path): File system location where to search for or install the dataset.
            info (DownloadInfo): Contains download information specific to either dataset or ground truth annotations.
        """
        if root.is_dir():
            logger.info("Found the %s folder.", info.description)
        else:
            root.mkdir(parents=True, exist_ok=True)

            logger.info("Downloading %s.", info.description)
            url = "http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal"
            zip_filepath = root / info.zip_filename
            with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=info.description) as progress_bar:
                urlretrieve(
                    url=f"{url}/{info.zip_filename}",
                    filename=zip_filepath,
                    reporthook=progress_bar.update_to,
                )
            logger.info("Checking hash")
            hash_check(zip_filepath, info.expected_hash)

            logger.info("Extracting the %s.", info.description)
            with zipfile.ZipFile(zip_filepath, "r") as zip_file:
                zip_file.extractall(root)
            # move contents to root
            for filename in (root / info.extracted_folder_name).glob("*"):
                move(str(filename), str(root / filename.name))
            (root / info.extracted_folder_name).rmdir()

            logger.info("Cleaning the zip file")
            (zip_filepath).unlink()
