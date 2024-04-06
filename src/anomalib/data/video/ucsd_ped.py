"""UCSD Pedestrian dataset."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from shutil import move
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib import TaskType
from anomalib.data.base import AnomalibVideoDataModule, AnomalibVideoDataset
from anomalib.data.base.video import VideoTargetFrame
from anomalib.data.utils import (
    DownloadInfo,
    Split,
    ValSplitMode,
    download_and_extract,
    read_image,
    read_mask,
    validate_path,
)
from anomalib.data.utils.video import ClipsIndexer

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

DOWNLOAD_INFO = DownloadInfo(
    name="UCSD Pedestrian",
    url="http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz",
    hashsum="2329af326951f5097fdd114c50e853957d3e569493a49d22fc082a9fd791915b",
)

CATEGORIES = ("UCSDped1", "UCSDped2")


def make_ucsd_dataset(path: Path, split: str | Split | None = None) -> DataFrame:
    """Create UCSD Pedestrian dataset by parsing the file structure.

    The files are expected to follow the structure:
        path/to/dataset/category/split/video_id/image_filename.tif
        path/to/dataset/category/split/video_id_gt/mask_filename.bmp

    Args:
        path (Path): Path to dataset
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
    path = validate_path(path)
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

    def get_mask(self, idx: int) -> np.ndarray | None:
        """Retrieve the masks from the file system."""
        video_idx, frames_idx = self.get_clip_location(idx)
        mask_folder = self.mask_paths[video_idx]
        if mask_folder == "":  # no gt masks available for this clip
            return None
        frames = self.clips[video_idx][frames_idx]

        mask_frames = sorted(Path(mask_folder).glob("*.bmp"))
        mask_paths = [mask_frames[idx] for idx in frames.int()]

        return torch.stack([read_mask(mask_path, as_tensor=True) for mask_path in mask_paths])

    def _compute_frame_pts(self) -> None:
        """Retrieve the number of frames in each video."""
        self.video_pts = []
        for video_path in self.video_paths:
            n_frames = len(list(Path(video_path).glob("*.tif")))
            self.video_pts.append(torch.Tensor(range(n_frames)))

        self.video_fps = [None] * len(self.video_paths)  # fps information cannot be inferred from folder structure

    def get_clip(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], int]:
        """Get a subclip from a list of videos.

        Args:
            idx (int): index of the subclip. Must be between 0 and num_clips().

        Returns:
            video (torch.Tensor)
            audio (torch.Tensor)
            info (dict)
            video_idx (int): index of the video in `video_paths`
        """
        if idx >= self.num_clips():
            msg = f"Index {idx} out of range ({self.num_clips()} number of clips)"
            raise IndexError(msg)
        video_idx, clip_idx = self.get_clip_location(idx)
        video_path = self.video_paths[video_idx]
        clip_pts = self.clips[video_idx][clip_idx]

        frames = sorted(Path(video_path).glob("*.tif"))

        frame_paths = [frames[pt] for pt in clip_pts.int()]
        video = torch.stack([read_image(frame_path, as_tensor=True) for frame_path in frame_paths])

        return video, torch.empty((1, 0)), {}, video_idx


class UCSDpedDataset(AnomalibVideoDataset):
    """UCSDped Dataset class.

    Args:
        task (TaskType): Task type, 'classification', 'detection' or 'segmentation'
        root (Path | str): Path to the root of the dataset
        category (str): Sub-category of the dataset, e.g. "UCSDped1" or "UCSDped2"
        split (str | Split | None): Split of the dataset, usually Split.TRAIN or Split.TEST
        clip_length_in_frames (int, optional): Number of video frames in each clip.
        frames_between_clips (int, optional): Number of frames between each consecutive video clip.
        target_frame (VideoTargetFrame): Specifies the target frame in the video clip, used for ground truth retrieval.
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
    """

    def __init__(
        self,
        task: TaskType,
        root: str | Path,
        category: str,
        split: Split,
        clip_length_in_frames: int = 2,
        frames_between_clips: int = 10,
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

        self.root_category = Path(root) / category
        self.split = split
        self.indexer_cls: Callable = UCSDpedClipsIndexer
        self.samples = make_ucsd_dataset(self.root_category, self.split)


class UCSDped(AnomalibVideoDataModule):
    """UCSDped DataModule class.

    Args:
        root (Path | str): Path to the root of the dataset
        category (str): Sub-category of the dataset, e.g. "UCSDped1" or "UCSDped2"
        clip_length_in_frames (int, optional): Number of video frames in each clip.
        frames_between_clips (int, optional): Number of frames between each consecutive video clip.
        target_frame (VideoTargetFrame): Specifies the target frame in the video clip, used for ground truth retrieval
        task (TaskType): Task type, 'classification', 'detection' or 'segmentation'
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
        root: Path | str = "./datasets/ucsd",
        category: str = "UCSDped2",
        clip_length_in_frames: int = 2,
        frames_between_clips: int = 10,
        target_frame: VideoTargetFrame = VideoTargetFrame.LAST,
        task: TaskType | str = TaskType.SEGMENTATION,
        image_size: tuple[int, int] | None = None,
        transform: Transform | None = None,
        train_transform: Transform | None = None,
        eval_transform: Transform | None = None,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
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
        self.category = category

        self.clip_length_in_frames = clip_length_in_frames
        self.frames_between_clips = frames_between_clips
        self.target_frame = VideoTargetFrame(target_frame)

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = UCSDpedDataset(
            task=self.task,
            transform=self.train_transform,
            clip_length_in_frames=self.clip_length_in_frames,
            frames_between_clips=self.frames_between_clips,
            target_frame=self.target_frame,
            root=self.root,
            category=self.category,
            split=Split.TRAIN,
        )

        self.test_data = UCSDpedDataset(
            task=self.task,
            transform=self.eval_transform,
            clip_length_in_frames=self.clip_length_in_frames,
            frames_between_clips=self.frames_between_clips,
            target_frame=self.target_frame,
            root=self.root,
            category=self.category,
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
