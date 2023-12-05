"""CUHK Avenue Dataset.

Description:
    This module provides a PyTorch Dataset and PyTorch Lightning DataModule for the CUHK Avenue dataset.
    If the dataset is not already present on the file system, the DataModule class will download and
    extract the dataset, converting the .mat mask files to .png format.

Reference:
    - Lu, Cewu, Jianping Shi, and Jiaya Jia. "Abnormal event detection at 150 fps in Matlab."
      In Proceedings of the IEEE International Conference on Computer Vision, 2013.
"""


# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

__all__ = ["Avenue", "AvenueDataset", "make_avenue_dataset"]

import logging
import math
from pathlib import Path
from shutil import move
from typing import TYPE_CHECKING

import albumentations as A  # noqa: N812
import cv2
import numpy as np
import scipy.io
from pandas import DataFrame

from anomalib.data.base import AnomalibVideoDataModule, AnomalibVideoDataset
from anomalib.data.base.video import VideoTargetFrame
from anomalib.data.utils import (
    DownloadInfo,
    InputNormalizationMethod,
    Split,
    ValSplitMode,
    download_and_extract,
    get_transforms,
)
from anomalib.data.utils.video import ClipsIndexer
from anomalib.utils.types import TaskType

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

DATASET_DOWNLOAD_INFO = DownloadInfo(
    name="Avenue Dataset",
    url="http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/Avenue_Dataset.zip",
    checksum="b7a34b212ecdd30efbd989a6dcb1aceb",
)
ANNOTATIONS_DOWNLOAD_INFO = DownloadInfo(
    name="Avenue Annotations",
    url="http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/ground_truth_demo.zip",
    checksum="e8e3bff99195b6b511534083b9dbe1f5",
)


def make_avenue_dataset(root: Path, gt_dir: Path, split: Split | str | None = None) -> DataFrame:
    """Create CUHK Avenue dataset by parsing the file structure.

    The files are expected to follow the structure:
        - path/to/dataset/[training_videos|testing_videos]/video_filename.avi
        - path/to/ground_truth/mask_filename.mat

    Args:
        root (Path): Path to dataset
        gt_dir (Path): Path to the ground truth
        split (Split | str | None = None, optional): Dataset split (ie., either train or test).
            Defaults to ``None``.

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
    """Clips class for Avenue dataset."""

    def get_mask(self, idx: int) -> np.ndarray | None:
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
        transform (A.Compose): Albumentations Compose object describing the transforms that are applied to the inputs.
        split (Split): Split of the dataset, usually Split.TRAIN or Split.TEST
        root (Path | str): Path to the root of the dataset
            Defaults to ``./datasets/avenue``.
        gt_dir (Path | str): Path to the ground truth files
            Defaults to ``./datasets/avenue/ground_truth_demo``.
        clip_length_in_frames (int, optional): Number of video frames in each clip.
            Defaults to ``2``.
        frames_between_clips (int, optional): Number of frames between each consecutive video clip.
            Defaults to ``1``.
        target_frame (VideoTargetFrame): Specifies the target frame in the video clip, used for ground truth retrieval
            Defaults to ``VideoTargetFrame.LAST``.

    Examples:
        To create an Avenue dataset to train a classification model:

        .. code-block:: python

            transform = A.Compose([A.Resize(256, 256), A.pytorch.ToTensorV2()])
            dataset = AvenueDataset(
                task="classification",
                transform=transform,
                split="train",
                root="./datasets/avenue/",
            )

            dataset.setup()
            dataset[0].keys()

            # Output: dict_keys(['image', 'video_path', 'frames', 'last_frame', 'original_image'])

        If you would like to test a segmentation model, you can use the following code:

        .. code-block:: python

            dataset = AvenueDataset(
                task="segmentation",
                transform=transform,
                split="test",
                root="./datasets/avenue/",
            )

            dataset.setup()
            dataset[0].keys()

            # Output: dict_keys(['image', 'mask', 'video_path', 'frames', 'last_frame', 'original_image', 'label'])

        Avenue video dataset can also be used as an image dataset if you set the clip length to 1. This means that each
        video frame will be treated as a separate sample. This is useful for training a classification model on the
        Avenue dataset. The following code shows how to create an image dataset for classification:

        .. code-block:: python

            dataset = AvenueDataset(
                task="classification",
                transform=transform,
                split="test",
                root="./datasets/avenue/",
                clip_length_in_frames=1,
            )

            dataset.setup()
            dataset[0].keys()
            # Output: dict_keys(['image', 'video_path', 'frames', 'last_frame', 'original_image', 'label'])

            dataset[0]["image"].shape
            # Output: torch.Size([3, 256, 256])
    """

    def __init__(
        self,
        task: TaskType,
        transform: A.Compose,
        split: Split,
        root: Path | str = "./datasets/avenue",
        gt_dir: Path | str = "./datasets/avenue/ground_truth_demo",
        clip_length_in_frames: int = 2,
        frames_between_clips: int = 1,
        target_frame: VideoTargetFrame = VideoTargetFrame.LAST,
    ) -> None:
        super().__init__(task, transform, clip_length_in_frames, frames_between_clips, target_frame)

        self.root = root if isinstance(root, Path) else Path(root)
        self.gt_dir = gt_dir if isinstance(gt_dir, Path) else Path(gt_dir)
        self.split = split
        self.indexer_cls: Callable = AvenueClipsIndexer

    def _setup(self) -> None:
        """Create and assign samples."""
        self.samples = make_avenue_dataset(self.root, self.gt_dir, self.split)


class Avenue(AnomalibVideoDataModule):
    """Avenue DataModule class.

    Args:
        root (Path | str): Path to the root  of the dataset
            Defaults to ``./datasets/avenue``.
        gt_dir (Path | str): Path to the ground truth files
            Defaults to ``./datasets/avenue/ground_truth_demo``.
        clip_length_in_frames (int, optional): Number of video frames in each clip.
            Defaults to ``2``.
        frames_between_clips (int, optional): Number of frames between each consecutive video clip.
            Defaults to ``1``.
        target_frame (VideoTargetFrame): Specifies the target frame in the video clip, used for ground truth retrieval
            Defaults to ``VideoTargetFrame.LAST``.
        task (TaskType): Task type, 'classification', 'detection' or 'segmentation'
            Defaults to ``TaskType.SEGMENTATION``.
        image_size (int | tuple[int, int] | None, optional): Size of the input image.
            Defaults to ``(256, 256)``.
        center_crop (int | tuple[int, int] | None, optional): When provided, the images will be center-cropped
            to the provided dimensions.
            Defaults to ``None``.
        normalization (InputNormalizationMethod | str): Normalization method to be applied to the input images.
            Defaults to ``InputNormalizationMethod.IMAGENET``.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Test batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            Defaults to ``8``.
        transform_config_train (str | A.Compose | None, optional): Config for pre-processing
            during training.
            Defaults to ``None``.
        transform_config_val (str | A.Compose | None, optional): Config for pre-processing
            during validation.
            Defaults to ``None``.
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
            Defaults to ``ValSplitMode.FROM_TEST``.
        val_split_ratio (float): Fraction of train or test images that will be reserved for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Seed which may be set to a fixed value for reproducibility.
            Defaults to ``None``.

    Examples:
        To create a DataModule for Avenue dataset with default parameters:

        .. code-block:: python

            datamodule = Avenue()
            datamodule.setup()

            i, data = next(enumerate(datamodule.train_dataloader()))
            data.keys()
            # Output: dict_keys(['image', 'video_path', 'frames', 'last_frame', 'original_image'])

            i, data = next(enumerate(datamodule.test_dataloader()))
            data.keys()
            # Output: dict_keys(['image', 'mask', 'video_path', 'frames', 'last_frame', 'original_image', 'label'])

            data["image"].shape
            # Output: torch.Size([32, 2, 3, 256, 256])

        Note that the default task type is segmentation and the dataloader returns a mask in addition to the input.
        Also, it is important to note that the dataloader returns a batch of clips, where each clip is a sequence of
        frames. The number of frames in each clip is determined by the ``clip_length_in_frames`` parameter. The
        ``frames_between_clips`` parameter determines the number of frames between each consecutive clip. The
        ``target_frame`` parameter determines which frame in the clip is used for ground truth retrieval. For example,
        if ``clip_length_in_frames=2``, ``frames_between_clips=1`` and ``target_frame=VideoTargetFrame.LAST``, then the
        dataloader will return a batch of clips where each clip contains two consecutive frames from the video. The
        second frame in each clip will be used as the ground truth for the first frame in the clip. The following code
        shows how to create a dataloader for classification:

        .. code-block:: python

            datamodule = Avenue(
                task="classification",
                clip_length_in_frames=2,
                frames_between_clips=1,
                target_frame=VideoTargetFrame.LAST
            )
            datamodule.setup()

            i, data = next(enumerate(datamodule.train_dataloader()))
            data.keys()
            # Output: dict_keys(['image', 'video_path', 'frames', 'last_frame', 'original_image'])

            data["image"].shape
            # Output: torch.Size([32, 2, 3, 256, 256])

    """

    def __init__(
        self,
        root: Path | str = "./datasets/avenue",
        gt_dir: Path | str = "./datasets/avenue/ground_truth_demo",
        clip_length_in_frames: int = 2,
        frames_between_clips: int = 1,
        target_frame: VideoTargetFrame = VideoTargetFrame.LAST,
        task: TaskType = TaskType.SEGMENTATION,
        image_size: int | tuple[int, int] = (256, 256),
        center_crop: int | tuple[int, int] | None = None,
        normalization: InputNormalizationMethod | str = InputNormalizationMethod.IMAGENET,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        transform_config_train: str | A.Compose | None = None,
        transform_config_eval: str | A.Compose | None = None,
        val_split_mode: ValSplitMode = ValSplitMode.FROM_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
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

        self.train_data = AvenueDataset(
            task=task,
            transform=transform_train,
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            target_frame=target_frame,
            root=root,
            gt_dir=gt_dir,
            split=Split.TRAIN,
        )

        self.test_data = AvenueDataset(
            task=task,
            transform=transform_eval,
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            target_frame=target_frame,
            root=root,
            gt_dir=gt_dir,
            split=Split.TEST,
        )

    def prepare_data(self) -> None:
        """Download the dataset if not available.

        This method checks if the specified dataset is available in the file system.
        If not, it downloads and extracts the dataset into the appropriate directory.

        Example:
            Assume the dataset is not available on the file system.
            Here's how the directory structure looks before and after calling the
            `prepare_data` method:

            Before:

            .. code-block:: bash

                $ tree datasets
                datasets
                ├── dataset1
                └── dataset2

            Calling the method:

            .. code-block:: python

                >> datamodule = Avenue()
                >> datamodule.prepare_data()

            After:

            .. code-block:: bash

                $ tree datasets
                datasets
                ├── dataset1
                ├── dataset2
                └── avenue
                    ├── ground_truth_demo
                    │   ├── ground_truth_show.m
                    │   ├── Readme.txt
                    │   ├── testing_label_mask
                    │   └── testing_videos
                    ├── testing_videos
                    │   ├── ...
                    │   └── 21.avi
                    ├── testing_vol
                    │   ├── ...
                    │   └── vol21.mat
                    ├── training_videos
                    │   ├── ...
                    │   └── 16.avi
                    └── training_vol
                        ├── ...
                        └── vol16.mat
        """
        if self.root.is_dir():
            logger.info("Found the dataset.")
        else:
            download_and_extract(self.root, DATASET_DOWNLOAD_INFO)
            download_and_extract(self.gt_dir, ANNOTATIONS_DOWNLOAD_INFO)

            # move contents to root
            folder_names = ["Avenue Dataset", "ground_truth_demo"]
            for root, folder_name in zip([self.root, self.gt_dir], folder_names, strict=True):
                extracted_folder = root / folder_name
                for filename in extracted_folder.glob("*"):
                    move(str(filename), str(root / filename.name))
                extracted_folder.rmdir()

            # convert masks
            self._convert_masks(self.gt_dir)

    @staticmethod
    def _convert_masks(gt_dir: Path) -> None:
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
            for mat_file, mask_folder in zip(mat_files, mask_folders, strict=True):
                mat = scipy.io.loadmat(mat_file)
                mask_folder.mkdir(parents=True, exist_ok=True)
                masks = mat["volLabel"].squeeze()
                for idx, mask in enumerate(masks):
                    filename = (mask_folder / str(idx).zfill(int(math.log10(len(masks)) + 1))).with_suffix(".png")
                    cv2.imwrite(str(filename), mask)
