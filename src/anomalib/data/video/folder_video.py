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
    read_image,
    read_mask,
    validate_path,
    video,
)
from anomalib.data.utils.path import _prepare_files_labels, validate_and_resolve_path
from anomalib.data.utils.video import ClipsIndexer, convert_video

logger = logging.getLogger(__name__)


# get avi and mp4 2 work
# ground truth 2 be npy and pt
# ground truth as a image folder
# get from most common extension an or functionality to extract the tensors

#    normal_dir: str | Path | Sequence[str | Path],
#    root: str | Path | None = None,
#    abnormal_dir: str | Path | Sequence[str | Path] | None = None,
#    normal_test_dir: str | Path | Sequence[str | Path] | None = None,
#    mask_dir: str | Path | Sequence[str | Path] | None = None,
#    split: str | Split | None = None,
#    extensions: tuple[str, ...] | None = None,


def make_folder_video_dataset(
    root: str | Path | None = None,
    normal_dir: str | Path | Sequence[str | Path] | None = None,
    mask_dir: str | Path | Sequence[str | Path] | None = None,
    test_dir: str | Path | Sequence[str | Path] | None = None,
    split: str | Split | None = None,
    extensions: tuple[str, ...] | None = None,
) -> DataFrame:
    """Make Folder Video Dataset.

    Args:
        normal_dir (str | Path | Sequence): Path to the directory containing normal images.
        root (str | Path | None): Path to the root directory of the dataset.
            Defaults to ``None``.
        abnormal_dir (str | Path | Sequence | None, optional): Path to the directory containing abnormal images.
            Defaults to ``None``.
        normal_test_dir (str | Path | Sequence | None, optional): Path to the directory containing normal images for
            the test dataset. Normal test images will be a split of `normal_dir` if `None`.
            Defaults to ``None``.
        mask_dir (str | Path | Sequence | None, optional): Path to the directory containing the mask annotations.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split (ie., Split.FULL, Split.TRAIN or Split.TEST).
            Defaults to ``None``.
        extensions (tuple[str, ...] | None, optional): Type of the image extensions to read from the directory.
            Defaults to ``None``.

    Returns:
        DataFrame: an output dataframe containing samples for the requested split (ie., train or test).

    Example:
        The following example shows how to get testing samples from ShanghaiTech dataset:

        >>> root = Path('./myDataset')
        >>> samples = make_folder_video_dataset(path, split='test')
        >>> samples.head()
            root            image_path                      split   mask_path
        0	mydataset	mydataset/testing/frames/01_0014	test	mydataset/testing/test_pixel_mask/01_0014.npy
        1	mydataset	mydataset/testing/frames/01_0015	test	mydataset/testing/test_pixel_mask/01_0015.npy
        ...

    Returns:
        DataFrame: an output dataframe containing samples for the requested split (ie., train or test)
    """

    def _resolve_path_and_convert_to_list(path: str | Path | Sequence[str | Path] | None) -> list[Path]:
        """Convert path to list of paths.

        Args:
            path (str | Path | Sequence | None): Path to replace with Sequence[str | Path].

        Examples:
            >>> _resolve_path_and_convert_to_list("dir")
            [Path("path/to/dir")]
            >>> _resolve_path_and_convert_to_list(["dir1", "dir2"])
            [Path("path/to/dir1"), Path("path/to/dir2")]

        Returns:
            list[Path]: The result of path replaced by Sequence[str | Path].
        """
        if isinstance(path, Sequence) and not isinstance(path, str):
            return [validate_and_resolve_path(dir_path, root) for dir_path in path]
        return [validate_and_resolve_path(path, root)] if path is not None else []

    # TODO(Bepitic): Get a mask for the testing is mandatory ?
    # but the mask for the training is optional
    # get paths to training videos
    # normal_dir = _resolve_path_and_convert_to_list(normal_dir)
    # test_dir = _resolve_path_and_convert_to_list(test_dir)
    # mask_dir = _resolve_path_and_convert_to_list(mask_dir)
    # if len(normal_dir) == 0:
    #     msg = "A folder location must be provided in normal_dir."
    #     raise ValueError(msg)

    # filenames = []
    # labels = []
    # dirs = {DirType.NORMAL: normal_dir}

    # if test_dir:
    #     dirs[DirType.ABNORMAL] = test_dir

    # if mask_dir:
    #     dirs[DirType.MASK] = mask_dir

    # TODO(bepitic): reflect on the true path in example and doc

    # # get paths to testing folders
    # path_list = [(str(root),) + filename.parts[-2:] for filename in normal_dir.glob("*.avi")]
    # print(path_list)
    # samples = DataFrame(
    #     path_list,
    #     columns=["root", "folder", "image_path"],
    # )  # TODO(Bepitic): Change image Path to video path ?
    # samples["split"] = split

    # samples["mask_path"] = ""
    # # TODO(Bepitic): Maybe other formats?
    # samples.loc[samples.root == str(path), "mask_path"] = mask_dir
    # # samples.loc[samples.root == str(path), "mask_path"] = (
    # # str(mask_dir) + "/" + samples.image_path.str.split(".").str[0] + ".npy"
    # # )

    # # TODO(Bepitic): Make a system to link both gt and datapoint into the same spot
    # samples["image_path"] = samples.root + "/" + samples.image_path

    # samples_list = [(str(root),) + filename.parts[-2:] for filename in root.glob("**/*.avi")]

    def _contains_files(path: Path, extensions: list) -> bool:
        """Check if the path contains at least one file with a given extension in the directory or its one-level subdir.

        Args:
        path (Path): The path to the folder.
        extensions (list): A list of file extensions to check for.

        Returns:
        bool: True if there is at least one file with a given extension, False otherwise.
        """
        for item in path.iterdir():
            if (
                item.is_file()
                and item.suffix in extensions
                or item.is_dir()
                and any(file.suffix in extensions for file in item.iterdir() if file.is_file())
            ):
                return True
        return False

    def _extract_samples(root: Path, path: Path) -> list:
        print("_extra fun:")
        print(f"path: {str(path)}")

        print(f"glob.path: {str([a for a in path.glob('./*')])}")
        samples_list = []
        if video.most_common_extension(path) in [".avi", ".mp4"]:
            print("_extra fun: a")

            most_common_extension = video.most_common_extension(path)
            samples_list.extend(
                [(str(root),) + filename.parts[-2:] for filename in sorted(path.glob(f"./*{most_common_extension}"))],
            )
            print(samples_list)
        elif video.most_common_extension(path) in [".png", ".tiff", ".bmp", ".tif"]:
            print("_extra fun: b")
            samples_list.extend(
                [
                    (str(root),) + filename.parts[-2:]
                    for filename in sorted(path.glob("./*"))
                    if _contains_files(path=path, extensions=[".png", ".tiff", ".bmp", ".tif"])
                ],
            )
        return samples_list

    root = validate_path(root)
    normal_dir = validate_path(root / normal_dir)
    normal_dir_ext = video.most_common_extension(normal_dir)
    test_dir = validate_path(root / test_dir)
    test_dir_ext = video.most_common_extension(test_dir)
    mask_dir = validate_path(root / mask_dir)
    mask_dir_ext = video.most_common_extension(mask_dir)
    print(_extract_samples(root, normal_dir))
    samples_list = []
    samples_list.extend(_extract_samples(root, normal_dir))

    print("sampleslist")
    print(samples_list)
    samples_list.extend(_extract_samples(root, test_dir))

    print("sampleslist")
    print(samples_list)
    samples_list_labels = []
    samples_list_labels.extend(
        [
            filename.parts[-1]
            for filename in sorted(mask_dir.glob("./*"))
            if _contains_files(path=mask_dir, extensions=[".avi", ".mp4", ".png", ".tiff", ".tif", ".bmp"])
        ],
    )

    print("sampleslist")
    print(samples_list)
    samples = DataFrame(samples_list, columns=["root", "folder", "image_path"])
    print(samples.head())

    # Remove DS_Strore
    samples = samples[~samples.loc[:, "image_path"].str.contains(".DS_Store")]
    print(samples.loc[samples.folder == normal_dir.parts[-1]].head())

    samples["image_path"] = samples.root + "/" + samples.folder + "/" + samples.image_path

    samples.loc[samples.folder == normal_dir.parts[-1], "split"] = "train"
    samples.loc[samples.folder == test_dir.parts[-1], "split"] = "test"
    samples_list_labels = [item for item in samples_list_labels if item != ".DS_Store"]
    print(samples_list_labels)
    print(samples.loc[samples.folder == test_dir.parts[-1]])
    samples.loc[samples.folder == test_dir.parts[-1], "mask_path"] = samples_list_labels
    samples.loc[samples.folder == normal_dir.parts[-1], "mask_path"] = ""

    if split:
        samples = samples[samples.split == split]
        samples = samples.reset_index(drop=True)
    print(samples.head())

    return samples


class FolderClipsIndexerVideo(ClipsIndexer):
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


class FolderClipsIndexerImgFrames(ClipsIndexer):
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

    def get_clip(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, any], int]:
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
        root: Path | str,
        mask_dir: Path | str,
        normal_dir: Path | str,
        test_dir: Path | str,
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

        self.root = Path(root)
        self.split = split
        self.mask_dir = mask_dir
        self.normal_dir = normal_dir
        self.test_dir = test_dir

        check_path = root

        if split == "test":
            check_path /= mask_dir
        else:
            check_path /= normal_dir

        if video.most_common_extension(check_path) in [".avi", ".mp4"]:
            self.indexer_cls = FolderClipsIndexerVideo
        elif video.most_common_extension(check_path) in [".bmp", ".png", ".tiff", ".tif"]:
            self.indexer_cls = FolderClipsIndexerImgFrames

        self.samples = make_folder_video_dataset(
            root=self.root,
            normal_dir=self.normal_dir,
            test_dir=self.test_dir,
            mask_dir=self.mask_dir,
            split=self.split,
        )


class FolderVideo(AnomalibVideoDataModule):
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
        normal_dir: Path | str,
        mask_dir: Path | str,
        test_dir: Path | str,
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
        self.normal_dir = Path(normal_dir)
        self.test_dir = Path(test_dir)
        self.mask_dir = Path(mask_dir)

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
            mask_dir=self.mask_dir,
            normal_dir=self.normal_dir,
            test_dir=self.test_dir,
            split=Split.TRAIN,
        )

        self.test_data = FolderDataset(
            task=self.task,
            transform=self.train_transform,
            clip_length_in_frames=self.clip_length_in_frames,
            frames_between_clips=self.frames_between_clips,
            target_frame=self.target_frame,
            root=self.root,
            mask_dir=self.mask_dir,
            normal_dir=self.normal_dir,
            test_dir=self.test_dir,
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
