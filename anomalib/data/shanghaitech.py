"""ShanghaiTech Campus Dataset.

Description:
    This module contains PyTorch Dataset and PyTorch
        Lightning DataModule for the ShanghaiTech Campus dataset.
    If the dataset is not on the file system, the DataModule class downloads and
        extracts the dataset and converts video files to a format that is readable by pyav.
Reference:
    - W. Liu and W. Luo, D. Lian and S. Gao. "Future Frame Prediction for Anomaly Detection -- A New Baseline."
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2018.
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path
from shutil import move
from typing import Any

import albumentations as A
import numpy as np
import pandas as pd
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
from anomalib.data.utils.video import ClipsIndexer, convert_video

logger = logging.getLogger(__name__)

DATASET_DOWNLOAD_INFO = DownloadInfo(
    name="ShanghaiTech Dataset",
    url="http://101.32.75.151:8181/dataset/shanghaitech.tar.gz",
    hash="08494decd30fb0fa213b519a9c555040",
)


def make_shanghaitech_dataset(root: Path, scene: int, split: Split | str | None = None) -> DataFrame:
    """Create ShanghaiTech dataset by parsing the file structure.

    The files are expected to follow the structure:
        path/to/dataset/[training_videos|testing_videos]/video_filename.avi
        path/to/ground_truth/mask_filename.mat

    Args:
        root (Path): Path to dataset
        scene (int): Index of the dataset scene (category) in range [1, 13]
        split (Split | str | None, optional): Dataset split (ie., either train or test). Defaults to None.

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
    scene_prefix = str(scene).zfill(2)

    # get paths to training videos
    train_root = Path(root) / "training/converted_videos"
    train_list = [(str(train_root),) + filename.parts[-2:] for filename in train_root.glob(f"{scene_prefix}_*.avi")]
    train_samples = DataFrame(train_list, columns=["root", "folder", "image_path"])
    train_samples["split"] = "train"

    # get paths to testing folders
    test_root = Path(root) / "testing/frames"
    test_folders = [filename for filename in sorted(test_root.glob(f"{scene_prefix}_*")) if filename.is_dir()]
    test_folders = [folder for folder in test_folders if len(list(folder.glob("*.jpg"))) > 0]
    test_list = [(str(test_root),) + folder.parts[-2:] for folder in test_folders]
    test_samples = DataFrame(test_list, columns=["root", "folder", "image_path"])
    test_samples["split"] = "test"

    samples = pd.concat([train_samples, test_samples], ignore_index=True)

    gt_root = Path(root) / "testing/test_pixel_mask"
    samples["mask_path"] = ""
    samples.loc[samples.root == str(test_root), "mask_path"] = (
        str(gt_root) + "/" + samples.image_path.str.split(".").str[0] + ".npy"
    )

    samples["image_path"] = samples.root + "/" + samples.image_path

    if split:
        samples = samples[samples.split == split]
        samples = samples.reset_index(drop=True)

    return samples


class ShanghaiTechTrainClipsIndexer(ClipsIndexer):
    """Clips indexer for ShanghaiTech dataset.

    The train and test subsets of the ShanghaiTech dataset use different file formats, so separate
    clips indexer implementations are needed.
    """

    def get_mask(self, idx: int) -> Tensor | None:
        """No masks available for training set."""
        return None


class ShanghaiTechTestClipsIndexer(ClipsIndexer):
    """Clips indexer for the test set of the ShanghaiTech Campus dataset.

    The train and test subsets of the ShanghaiTech dataset use different file formats, so separate
    clips indexer implementations are needed.
    """

    def get_mask(self, idx: int) -> Tensor | None:
        """Retrieve the masks from the file system."""

        video_idx, frames_idx = self.get_clip_location(idx)
        mask_file = self.mask_paths[video_idx]
        if mask_file == "":  # no gt masks available for this clip
            return None
        frames = self.clips[video_idx][frames_idx]

        vid_masks = np.load(mask_file)
        masks = np.take(vid_masks, frames, 0)
        return masks

    def _compute_frame_pts(self) -> None:
        """Retrieve the number of frames in each video."""
        self.video_pts = []
        for video_path in self.video_paths:
            n_frames = len(list(Path(video_path).glob("*.jpg")))
            self.video_pts.append(Tensor(range(n_frames)))

        self.video_fps = [None] * len(self.video_paths)  # fps information cannot be inferred from folder structure

    def get_clip(self, idx: int) -> tuple[Tensor, Tensor, dict[str, Any], int]:
        """Gets a subclip from a list of videos.

        Args:
            idx (int): index of the subclip. Must be between 0 and num_clips().

        Returns:
            video (Tensor)
            audio (Tensor)
            info (Dict)
            video_idx (int): index of the video in `video_paths`
        """
        if idx >= self.num_clips():
            raise IndexError(f"Index {idx} out of range ({self.num_clips()} number of clips)")
        video_idx, clip_idx = self.get_clip_location(idx)
        video_path = self.video_paths[video_idx]
        clip_pts = self.clips[video_idx][clip_idx]

        frames = sorted(Path(video_path).glob("*.jpg"))

        frame_paths = [frames[pt] for pt in clip_pts.int()]
        video = torch.stack([Tensor(read_image(str(frame_path))) for frame_path in frame_paths])

        return video, torch.empty((1, 0)), {}, video_idx


class ShanghaiTechDataset(AnomalibVideoDataset):
    """ShanghaiTech Dataset class.

    Args:
        task (TaskType): Task type, 'classification', 'detection' or 'segmentation'
        root (Path | str): Path to the root of the dataset
        scene (int): Index of the dataset scene (category) in range [1, 13]
        transform (A.Compose): Albumentations Compose object describing the transforms that are applied to the inputs.
        split (Split): Split of the dataset, usually Split.TRAIN or Split.TEST
        clip_length_in_frames (int, optional): Number of video frames in each clip.
        frames_between_clips (int, optional): Number of frames between each consecutive video clip.
    """

    def __init__(
        self,
        task: TaskType,
        root: Path | str,
        scene: int,
        transform: A.Compose,
        split: Split,
        clip_length_in_frames: int = 1,
        frames_between_clips: int = 1,
    ):
        super().__init__(task, transform, clip_length_in_frames, frames_between_clips)

        self.root = root
        self.scene = scene
        self.split = split
        self.indexer_cls = ShanghaiTechTrainClipsIndexer if self.split == Split.TRAIN else ShanghaiTechTestClipsIndexer

    def _setup(self):
        """Create and assign samples."""
        self.samples = make_shanghaitech_dataset(self.root, self.scene, self.split)


class ShanghaiTech(AnomalibVideoDataModule):
    """ShanghaiTech DataModule class.

    Args:
        root (Path | str): Path to the root of the dataset
        scene (int): Index of the dataset scene (category) in range [1, 13]
        clip_length_in_frames (int, optional): Number of video frames in each clip.
        frames_between_clips (int, optional): Number of frames between each consecutive video clip.
        task TaskType): Task type, 'classification', 'detection' or 'segmentation'
        image_size (int | tuple[int, int] | None, optional): Size of the input image.
            Defaults to None.
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
        scene: int,
        clip_length_in_frames: int = 1,
        frames_between_clips: int = 1,
        task: TaskType = TaskType.SEGMENTATION,
        image_size: int | tuple[int, int] | None = None,
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
        self.scene = scene

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

        self.train_data = ShanghaiTechDataset(
            task=task,
            transform=transform_train,
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            root=root,
            scene=scene,
            split=Split.TRAIN,
        )

        self.test_data = ShanghaiTechDataset(
            task=task,
            transform=transform_eval,
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            root=root,
            scene=scene,
            split=Split.TEST,
        )

    def prepare_data(self) -> None:
        """Download the dataset and convert video files."""
        training_root = self.root / "training"
        if training_root.is_dir():
            logger.info("Found the dataset.")
        else:
            download_and_extract(self.root, DATASET_DOWNLOAD_INFO)

            # move contents to root
            extracted_folder = self.root / "shanghaitech"
            for filename in extracted_folder.glob("*"):
                move(str(filename), str(self.root / filename.name))
            extracted_folder.rmdir()

        # convert images if not done already
        vid_dir = training_root / "videos"
        converted_vid_dir = training_root / "converted_videos"
        vid_count = len(list(vid_dir.glob("*")))
        converted_vid_count = len(list(converted_vid_dir.glob("*")))
        if not vid_count == converted_vid_count:
            self._convert_training_videos(vid_dir, converted_vid_dir)

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
        training_videos = sorted(list(video_folder.glob("*")))
        for video_idx, video_path in enumerate(training_videos):
            logger.info("Converting training video %s (%i/%i)...", video_path.name, video_idx + 1, len(training_videos))
            file_name = video_path.name
            target_path = target_folder / file_name
            convert_video(video_path, target_path, codec="XVID")
