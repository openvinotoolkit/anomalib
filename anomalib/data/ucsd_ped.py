"""UCSD Pedestrian dataset."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path
from shutil import move
from typing import Any, Callable

import albumentations as A
import cv2
import numpy as np
import torch
from pandas import DataFrame
from torch import Tensor

from anomalib.data.base import AnomalibVideoDataModule, AnomalibVideoDataset
from anomalib.data.task_type import TaskType
from anomalib.data.utils import (
    DownloadInfo,
    InputNormalizationMethod,
    Split,
    ValSplitMode,
    download_and_extract,
    get_transforms,
    read_image,
)
from anomalib.data.utils.video import ClipsIndexer

logger = logging.getLogger(__name__)

DOWNLOAD_INFO = DownloadInfo(
    name="UCSD Pedestrian",
    url="http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz",
    hash="5006421b89885f45a6f93b041145f2eb",
)


def make_ucsd_dataset(path: Path, split: str | Split | None = None) -> DataFrame:
    """Create UCSD Pedestrian dataset by parsing the file structure.

    The files are expected to follow the structure:
        path/to/dataset/category/split/video_id/image_filename.tif
        path/to/dataset/category/split/video_id_gt/mask_filename.bmp

    Args:
        root (Path): Path to dataset
        split (str | Split | None, optional): Dataset split (ie., either train or test). Defaults to None.

    Example:
        The following example shows how to get testing samples from UCSDped2 category:

        >>> root = Path('./UCSDped')
        >>> category = 'UCSDped2'
        >>> path = root / category
        >>> path
        PosixPath('UCSDped/UCSDped2')

        >>> samples = make_ucsd_dataset(path, split='test')
        >>> samples.head()
           root             folder image_path                    mask_path                         split
        0  UCSDped/UCSDped2 Test   UCSDped/UCSDped2/Test/Test001 UCSDped/UCSDped2/Test/Test001_gt  test
        1  UCSDped/UCSDped2 Test   UCSDped/UCSDped2/Test/Test002 UCSDped/UCSDped2/Test/Test002_gt  test
        ...

    Returns:
        DataFrame: an output dataframe containing samples for the requested split (ie., train or test)
    """
    folders = [filename for filename in sorted(path.glob("*/*")) if filename.is_dir()]
    folders = [folder for folder in folders if list(folder.glob("*.tif"))]

    samples_list = [(str(path),) + folder.parts[-2:] for folder in folders]
    samples = DataFrame(samples_list, columns=["root", "folder", "image_path"])

    samples.loc[samples.folder == "Test", "mask_path"] = samples.image_path.str.split(".").str[0] + "_gt"
    samples.loc[samples.folder == "Test", "mask_path"] = samples.root + "/" + samples.folder + "/" + samples.mask_path
    samples.loc[samples.folder == "Train", "mask_path"] = ""

    samples["image_path"] = samples.root + "/" + samples.folder + "/" + samples.image_path

    samples.loc[samples.folder == "Train", "split"] = "train"
    samples.loc[samples.folder == "Test", "split"] = "test"

    if split:
        samples = samples[samples.split == split]
        samples = samples.reset_index(drop=True)

    return samples


class UCSDpedClipsIndexer(ClipsIndexer):
    """Clips class for UCSDped dataset."""

    def get_mask(self, idx) -> np.ndarray | None:
        """Retrieve the masks from the file system."""

        video_idx, frames_idx = self.get_clip_location(idx)
        mask_folder = self.mask_paths[video_idx]
        if mask_folder == "":  # no gt masks available for this clip
            return None
        frames = self.clips[video_idx][frames_idx]

        mask_frames = sorted(Path(mask_folder).glob("*.bmp"))
        mask_paths = [mask_frames[idx] for idx in frames.int()]

        masks = np.stack([cv2.imread(str(mask_path), flags=0) / 255.0 for mask_path in mask_paths])
        return masks

    def _compute_frame_pts(self) -> None:
        """Retrieve the number of frames in each video."""
        self.video_pts = []
        for video_path in self.video_paths:
            n_frames = len(list(Path(video_path).glob("*.tif")))
            self.video_pts.append(Tensor(range(n_frames)))

        self.video_fps = [None] * len(self.video_paths)  # fps information cannot be inferred from folder structure

    def get_clip(self, idx: int) -> tuple[Tensor, Tensor, dict[str, Any], int]:
        """Gets a subclip from a list of videos.

        Args:
            idx (int): index of the subclip. Must be between 0 and num_clips().

        Returns:
            video (Tensor)
            audio (Tensor)
            info (dict)
            video_idx (int): index of the video in `video_paths`
        """
        if idx >= self.num_clips():
            raise IndexError(f"Index {idx} out of range ({self.num_clips()} number of clips)")
        video_idx, clip_idx = self.get_clip_location(idx)
        video_path = self.video_paths[video_idx]
        clip_pts = self.clips[video_idx][clip_idx]

        frames = sorted(Path(video_path).glob("*.tif"))

        frame_paths = [frames[pt] for pt in clip_pts.int()]
        video = torch.stack([Tensor(read_image(str(frame_path))) for frame_path in frame_paths])

        return video, torch.empty((1, 0)), {}, video_idx


class UCSDpedDataset(AnomalibVideoDataset):
    """UCSDped Dataset class.

    Args:
        task (TaskType): Task type, 'classification', 'detection' or 'segmentation'
        root (Path | str): Path to the root of the dataset
        category (str): Sub-category of the dataset, e.g. 'bottle'
        transform (A.Compose): Albumentations Compose object describing the transforms that are applied to the inputs.
        split (str | Split | None): Split of the dataset, usually Split.TRAIN or Split.TEST
        clip_length_in_frames (int, optional): Number of video frames in each clip.
        frames_between_clips (int, optional): Number of frames between each consecutive video clip.
    """

    def __init__(
        self,
        task: TaskType,
        root: str | Path,
        category: str,
        transform: A.Compose,
        split: Split,
        clip_length_in_frames: int = 1,
        frames_between_clips: int = 1,
    ) -> None:
        super().__init__(task, transform, clip_length_in_frames, frames_between_clips)

        self.root_category = Path(root) / category
        self.split = split
        self.indexer_cls: Callable = UCSDpedClipsIndexer

    def _setup(self) -> None:
        """Create and assign samples."""
        self.samples = make_ucsd_dataset(self.root_category, self.split)


class UCSDped(AnomalibVideoDataModule):
    """UCSDped DataModule class.

    Args:
        root (Path | str): Path to the root of the dataset
        category (str): Sub-category of the dataset, e.g. 'bottle'
        clip_length_in_frames (int, optional): Number of video frames in each clip.
        frames_between_clips (int, optional): Number of frames between each consecutive video clip.
        task (TaskType): Task type, 'classification', 'detection' or 'segmentation'
        image_size (int | tuple[int, int] | None, optional): Size of the input image.
            Defaults to None.
        center_crop (int | tuple[int, int] | None, optional): When provided, the images will be center-cropped
            to the provided dimensions.
        normalize (bool): When True, the images will be normalized to the ImageNet statistics.
        center_crop (int | tuple[int, int] | None, optional): When provided, the images will be center-cropped
            to the provided dimensions.
        normalize (bool): When True, the images will be normalized to the ImageNet statistics.
        train_batch_size (int, optional): Training batch size. Defaults to 32.
        eval_batch_size (int, optional): Test batch size. Defaults to 32.
        num_workers (int, optional): Number of workers. Defaults to 8.
        transform_config_train (str | A.Compose | None, optional): Config for pre-processing
            during training.
            Defaults to None.
        transform_config_val (str | A.Compose | None, optional): Config for pre-processing
            during validation.
            Defaults to None.
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
        val_split_ratio (float): Fraction of train or test images that will be reserved for validation.
        seed (int | None, optional): Seed which may be set to a fixed value for reproducibility.
    """

    def __init__(
        self,
        root: Path | str,
        category: str,
        clip_length_in_frames: int = 1,
        frames_between_clips: int = 1,
        task: TaskType = TaskType.SEGMENTATION,
        image_size: int | tuple[int, int] | None = None,
        center_crop: int | tuple[int, int] | None = None,
        normalization: str | InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
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
        self.category = category

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

        self.train_data = UCSDpedDataset(
            task=task,
            transform=transform_train,
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            root=root,
            category=category,
            split=Split.TRAIN,
        )

        self.test_data = UCSDpedDataset(
            task=task,
            transform=transform_eval,
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            root=root,
            category=category,
            split=Split.TEST,
        )

    def prepare_data(self) -> None:
        """Download the dataset if not available."""
        if (self.root / self.category).is_dir():
            logger.info("Found the dataset.")
        else:
            download_and_extract(self.root, DOWNLOAD_INFO)

            # move contents to root
            extracted_folder = self.root / "UCSD_Anomaly_Dataset.v1p2"
            for filename in extracted_folder.glob("*"):
                move(str(filename), str(self.root / filename.name))
            extracted_folder.rmdir()
