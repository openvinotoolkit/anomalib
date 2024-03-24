"""Custom video Folder Dataset.

This script creates a custom dataset from a folder.
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib import TaskType
from anomalib.data.base import AnomalibVideoDataModule, AnomalibVideoDataset
from anomalib.data.base.video import VideoTargetFrame
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import (
    DirType,
    LabelName,
    Split,
    TestSplitMode,
    ValSplitMode,
    validate_path,
)
from anomalib.data.utils.path import _prepare_files_labels, validate_and_resolve_path
from anomalib.data.utils.video import ClipsIndexer, convert_video

logger = logging.getLogger(__name__)


def make_folder_video_dataset(
    path: Path | None = None,
    gt_dir: Path | None = None,
    split: str | Split | None = None,
) -> DataFrame:
    """Make Folder Video Dataset.

    Args:
        path (Path | None): Path to the video directory of the dataset(.avi).
            Defaults to ``None``.
        gt_dir (Path | None, optional): Path to the directory containing the mask annotations(.npy).
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split (ie., Split.FULL, Split.TRAIN or Split.TEST).
            Defaults to ``None``.
        extensions (tuple[str, ...] | None, optional): Type of the image extensions to read from the directory.
            Defaults to ``None``.

    Returns:
        DataFrame: an output dataframe containing samples for the requested split (ie., train or test).

    Example:
        The following example shows how to get testing samples from ShanghaiTech dataset:

        >>> root = Path('./shanghaiTech')
        >>> scene = 1
        >>> samples = make_avenue_dataset(path, scene, split='test')
        >>> samples.head()
            root            image_path                          split   mask_path
        0	shanghaitech	shanghaitech/testing/frames/01_0014	test	shanghaitech/testing/test_pixel_mask/01_0014.npy
        1	shanghaitech	shanghaitech/testing/frames/01_0015	test	shanghaitech/testing/test_pixel_mask/01_0015.npy
        ...

    Returns:
        DataFrame: an output dataframe containing samples for the requested split (ie., train or test)
    """
    # TODO(Bepitic): Get a mask for the testing is mandatory ?
    # but the mask for the training is optional
    # get paths to training videos
    path = validate_path(path)
    # TODO(bepitic): reflect on the true path in example and doc

    # get paths to testing folders
    path_list = [(str(path),) + filename.parts[-2:] for filename in path.glob("*.avi")]
    samples = DataFrame(
        path_list,
        columns=["root", "folder", "image_path"],
    )  # TODO(Bepitic): Change image Path to video path ?
    samples["split"] = split

    samples["mask_path"] = ""
    # TODO(Bepitic): Maybe other formats?
    samples.loc[samples.root == str(path), "mask_path"] = (
        str(gt_dir) + "/" + samples.image_path.str.split(".").str[0] + ".npy"
    )

    # TODO(Bepitic): Make a system to link both gt and datapoint into the same spot
    samples["image_path"] = samples.root + "/" + samples.image_path

    return samples


class FolderClipsIndexer(ClipsIndexer):
    """Clips indexer for the test set Folder video dataset."""

    def get_mask(self, idx: int) -> torch.Tensor | None:
        """Retrieve the masks from the file system."""
        video_idx, frames_idx = self.get_clip_location(idx)
        mask_file = self.mask_paths[video_idx]
        if mask_file == "":  # no gt masks available for this clip
            return None
        frames = self.clips[video_idx][frames_idx]

        vid_masks = np.load(mask_file)
        return torch.tensor(np.take(vid_masks, frames, 0))


class FolderDataset(AnomalibVideoDataset):
    """Folder Dataset class.

    Args:
        task (TaskType): Task type, 'classification', 'detection' or 'segmentation'
        split (Split): Split of the dataset, usually Split.TRAIN or Split.TEST
        path (Path | str): Path to the training/testing videos of the dataset (.avi)
        path_gt (Path | str): Path to the masks fror the training videos of the dataset (.npy)
        clip_length_in_frames (int, optional): Number of video frames in each clip.
        frames_between_clips (int, optional): Number of frames between each consecutive video clip.
        target_frame (VideoTargetFrame): Specifies the target frame in the video clip, used for ground truth retrieval.
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
    """

    def __init__(
        self,
        task: TaskType,
        split: Split,
        path: Path | str,
        path_gt: Path | str,
        clip_length_in_frames: int = 2,
        frames_between_clips: int = 1,
        target_frame: VideoTargetFrame = VideoTargetFrame.LAST,
        transform: Transform | None = None,
    ) -> None:
        super().__init__(
            task=task,
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            target_frame=target_frame,
            transform=transform,
        )

        self.path = Path(path)
        self.split = split
        self.path_gt = path_gt
        self.indexer_cls = FolderClipsIndexer
        self.samples = make_folder_video_dataset(
            self.path,
            self.path_gt,
            self.split,
        )  # TODO(Bepitic):Check the constructor


class Folder(AnomalibVideoDataModule):
    """Folder DataModule class.

    Args:
        root (Path | str): Path to the root of the dataset
        clip_length_in_frames (int, optional): Number of video frames in each clip.
        frames_between_clips (int, optional): Number of frames between each consecutive video clip.
        target_frame (VideoTargetFrame): Specifies the target frame in the video clip, used for ground truth retrieval
        task TaskType): Task type, 'classification', 'detection' or 'segmentation'
        image_size (tuple[int, int], optional): Size to which input images should be resized.
            Defaults to ``None``.
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
        train_transform (Transform, optional): Transforms that should be applied to the input images during training.
            Defaults to ``None``.
        eval_transform (Transform, optional): Transforms that should be applied to the input images during evaluation.
            Defaults to ``None``.
        train_batch_size (int, optional): Training batch size. Defaults to 32.
        eval_batch_size (int, optional): Test batch size. Defaults to 32.
        num_workers (int, optional): Number of workers. Defaults to 8.
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
        val_split_ratio (float): Fraction of train or test images that will be reserved for validation.
        seed (int | None, optional): Seed which may be set to a fixed value for reproducibility.
    """

    def __init__(
        self,
        root: Path | str,
        clip_length_in_frames: int = 2,
        frames_between_clips: int = 1,
        target_frame: VideoTargetFrame = VideoTargetFrame.LAST,
        task: TaskType = TaskType.SEGMENTATION,
        image_size: tuple[int, int] | None = None,
        transform: Transform | None = None,
        train_transform: Transform | None = None,
        eval_transform: Transform | None = None,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        val_split_mode: ValSplitMode = ValSplitMode.SAME_AS_TEST,
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
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        self.task = TaskType(task)
        self.root = Path(root)

        self.clip_length_in_frames = clip_length_in_frames
        self.frames_between_clips = frames_between_clips
        self.target_frame = target_frame

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = FolderDataset(
            task=self.task,
            transform=self.train_transform,
            clip_length_in_frames=self.clip_length_in_frames,
            frames_between_clips=self.frames_between_clips,
            target_frame=self.target_frame,
            root=self.root,
            scene=self.scene,
            split=Split.TRAIN,
        )

        self.test_data = FolderDataset(
            task=self.task,
            transform=self.eval_transform,
            clip_length_in_frames=self.clip_length_in_frames,
            frames_between_clips=self.frames_between_clips,
            target_frame=self.target_frame,
            root=self.root,
            scene=self.scene,
            split=Split.TEST,
        )

    @staticmethod
    def _convert_training_videos(video_folder: Path, target_folder: Path) -> None:
        """Re-code the training videos to ensure correct reading of frames by torchvision.

        The encoding of the raw video files in the ShanghaiTech dataset causes some problems when
        reading the frames using pyav. To prevent this, we read the frames from the video files using opencv,
        and write them to a new video file that can be parsed correctly with pyav.

        Args:
            video_folder (Path): Path to the folder of training videos.
            target_folder (Path): File system location where the converted videos will be stored.
        """
        training_videos = sorted(video_folder.glob("*"))
        for video_idx, video_path in enumerate(training_videos):
            logger.info("Converting training video %s (%i/%i)...", video_path.name, video_idx + 1, len(training_videos))
            file_name = video_path.name
            target_path = target_folder / file_name
            convert_video(video_path, target_path, codec="XVID")
