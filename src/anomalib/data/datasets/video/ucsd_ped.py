"""UCSD Pedestrian Dataset."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base.video import AnomalibVideoDataset, VideoTargetFrame
from anomalib.data.utils import Split, read_image, read_mask, validate_path
from anomalib.data.utils.video import ClipsIndexer

if TYPE_CHECKING:
    from collections.abc import Callable

CATEGORIES = ("UCSDped1", "UCSDped2")


class UCSDpedDataset(AnomalibVideoDataset):
    """UCSDped Dataset class.

    Args:
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
        root: str | Path,
        category: str,
        split: Split,
        clip_length_in_frames: int = 2,
        frames_between_clips: int = 10,
        target_frame: VideoTargetFrame = VideoTargetFrame.LAST,
        transform: Transform | None = None,
    ) -> None:
        super().__init__(
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            target_frame=target_frame,
            transform=transform,
        )

        self.root_category = Path(root) / category
        self.split = split
        self.indexer_cls: Callable = UCSDpedClipsIndexer
        self.samples = make_ucsd_dataset(self.root_category, self.split)


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

    # infer the task type
    samples.attrs["task"] = "classification" if (samples["mask_path"] == "").all() else "segmentation"

    if split:
        samples = samples[samples.split == split]
        samples = samples.reset_index(drop=True)

    return samples
