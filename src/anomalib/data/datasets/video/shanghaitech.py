"""ShanghaiTech Campus Dataset.

This module provides PyTorch Dataset implementation for the ShanghaiTech Campus
dataset for abnormal event detection. The dataset contains surveillance videos
with both normal and abnormal events.

If the dataset is not already present on the file system, the DataModule class
will download and extract the dataset, converting the video files to a format
readable by pyav.

The dataset expects the following directory structure::

    root/
    ├── training/
    │   └── converted_videos/
    │       ├── 01_001.avi
    │       ├── 01_002.avi
    │       └── ...
    └── testing/
        ├── frames/
        │   ├── 01_0014/
        │   │   ├── 000001.jpg
        │   │   └── ...
        │   └── ...
        └── test_pixel_mask/
            ├── 01_0014.npy
            └── ...

Example:
    Create a dataset for training:

    >>> from anomalib.data.datasets import ShanghaiTechDataset
    >>> from anomalib.data.utils import Split
    >>> dataset = ShanghaiTechDataset(
    ...     root="./datasets/shanghaitech",
    ...     scene=1,
    ...     split=Split.TRAIN
    ... )
    >>> dataset[0].keys()
    dict_keys(['image', 'video_path', 'frames', 'last_frame', 'original_image'])

    Create a test dataset:

    >>> dataset = ShanghaiTechDataset(
    ...     root="./datasets/shanghaitech",
    ...     scene=1,
    ...     split=Split.TEST
    ... )
    >>> dataset[0].keys()
    dict_keys(['image', 'mask', 'video_path', 'frames', 'last_frame',
    'original_image', 'label'])

License:
    ShanghaiTech Campus Dataset is released under the BSD 2-Clause License.

Reference:
    Liu, W., Luo, W., Lian, D., & Gao, S. (2018). Future frame prediction for
    anomaly detection--a new baseline. In Proceedings of the IEEE conference on
    computer vision and pattern recognition (pp. 6536-6545).
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base.video import AnomalibVideoDataset, VideoTargetFrame
from anomalib.data.utils import Split, read_image, validate_path
from anomalib.data.utils.video import ClipsIndexer


class ShanghaiTechDataset(AnomalibVideoDataset):
    """ShanghaiTech Dataset class.

    Args:
        split (Split): Dataset split - either ``Split.TRAIN`` or ``Split.TEST``
        root (Path | str): Path to the root directory containing the dataset.
            Defaults to ``"./datasets/shanghaitech"``.
        scene (int): Index of the dataset scene (category) in range [1, 13].
            Defaults to ``1``.
        clip_length_in_frames (int, optional): Number of frames in each video
            clip. Defaults to ``2``.
        frames_between_clips (int, optional): Number of frames between each
            consecutive video clip. Defaults to ``1``.
        target_frame (VideoTargetFrame): Specifies which frame in the clip to use
            for ground truth retrieval. Defaults to ``VideoTargetFrame.LAST``.
        augmentations (Transform, optional): Augmentations that should be applied to the input images.
            Defaults to ``None``.

    Example:
        >>> from anomalib.data.datasets import ShanghaiTechDataset
        >>> from anomalib.data.utils import Split
        >>> dataset = ShanghaiTechDataset(
        ...     root="./datasets/shanghaitech",
        ...     scene=1,
        ...     split=Split.TRAIN
        ... )
    """

    def __init__(
        self,
        split: Split,
        root: Path | str = "./datasets/shanghaitech",
        scene: int = 1,
        clip_length_in_frames: int = 2,
        frames_between_clips: int = 1,
        target_frame: VideoTargetFrame = VideoTargetFrame.LAST,
        augmentations: Transform | None = None,
    ) -> None:
        super().__init__(
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            target_frame=target_frame,
            augmentations=augmentations,
        )

        self.root = Path(root)
        self.scene = scene
        self.split = split
        self.indexer_cls = ShanghaiTechTrainClipsIndexer if self.split == Split.TRAIN else ShanghaiTechTestClipsIndexer
        self.samples = make_shanghaitech_dataset(self.root, self.scene, self.split)


class ShanghaiTechTrainClipsIndexer(ClipsIndexer):
    """Clips indexer for ShanghaiTech training dataset.

    The train and test subsets use different file formats, so separate clips
    indexer implementations are needed.
    """

    @staticmethod
    def get_mask(idx: int) -> torch.Tensor | None:
        """No masks available for training set.

        Args:
            idx (int): Index of the clip.

        Returns:
            None: Training set has no masks.
        """
        del idx  # Unused argument
        return None


class ShanghaiTechTestClipsIndexer(ClipsIndexer):
    """Clips indexer for ShanghaiTech test dataset.

    The train and test subsets use different file formats, so separate clips
    indexer implementations are needed.
    """

    def get_mask(self, idx: int) -> torch.Tensor | None:
        """Retrieve the masks from the file system.

        Args:
            idx (int): Index of the clip.

        Returns:
            torch.Tensor | None: Ground truth mask if available, else None.
        """
        video_idx, frames_idx = self.get_clip_location(idx)
        mask_file = self.mask_paths[video_idx]
        if mask_file == "":  # no gt masks available for this clip
            return None
        frames = self.clips[video_idx][frames_idx]

        vid_masks = np.load(mask_file)
        return torch.tensor(np.take(vid_masks, frames, 0))

    def _compute_frame_pts(self) -> None:
        """Retrieve the number of frames in each video."""
        self.video_pts = []
        for video_path in self.video_paths:
            n_frames = len(list(Path(video_path).glob("*.jpg")))
            self.video_pts.append(torch.Tensor(range(n_frames)))

        # fps information cannot be inferred from folder structure
        self.video_fps = [None] * len(self.video_paths)

    def get_clip(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], int]:
        """Get a subclip from a list of videos.

        Args:
            idx (int): Index of the subclip. Must be between 0 and num_clips().

        Returns:
            tuple containing:
                - video (torch.Tensor): Video clip tensor
                - audio (torch.Tensor): Empty audio tensor
                - info (dict): Empty info dictionary
                - video_idx (int): Index of the video in video_paths

        Raises:
            IndexError: If idx is out of range.
        """
        if idx >= self.num_clips():
            msg = f"Index {idx} out of range ({self.num_clips()} number of clips)"
            raise IndexError(msg)
        video_idx, clip_idx = self.get_clip_location(idx)
        video_path = self.video_paths[video_idx]
        clip_pts = self.clips[video_idx][clip_idx]

        frames = sorted(Path(video_path).glob("*.jpg"))

        frame_paths = [frames[pt] for pt in clip_pts.int()]
        video = torch.stack([read_image(frame_path, as_tensor=True) for frame_path in frame_paths])

        return video, torch.empty((1, 0)), {}, video_idx


def make_shanghaitech_dataset(root: Path, scene: int, split: Split | str | None = None) -> DataFrame:
    """Create ShanghaiTech dataset by parsing the file structure.

    The files are expected to follow the structure::

        root/
        ├── training/
        │   └── converted_videos/
        │       ├── 01_001.avi
        │       └── ...
        └── testing/
            ├── frames/
            │   ├── 01_0014/
            │   │   ├── 000001.jpg
            │   │   └── ...
            │   └── ...
            └── test_pixel_mask/
                ├── 01_0014.npy
                └── ...

    Args:
        root (Path): Path to dataset root directory.
        scene (int): Index of the dataset scene (category) in range [1, 13].
        split (Split | str | None, optional): Dataset split (train or test).
            Defaults to ``None``.

    Returns:
        DataFrame: DataFrame containing samples for the requested split.

    Example:
        >>> from pathlib import Path
        >>> root = Path('./shanghaitech')
        >>> scene = 1
        >>> samples = make_shanghaitech_dataset(root, scene, split='test')
        >>> samples.head()
            root         image_path                       split    mask_path
        0   shanghaitech shanghaitech/testing/frames/01_0014 test ...01_0014.npy
        1   shanghaitech shanghaitech/testing/frames/01_0015 test ...01_0015.npy
    """
    scene_prefix = str(scene).zfill(2)

    # get paths to training videos
    root = validate_path(root)
    train_root = root / "training/converted_videos"
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

    # infer the task type
    samples.attrs["task"] = "classification" if (samples["mask_path"] == "").all() else "segmentation"

    if split:
        samples = samples[samples.split == split]
        samples = samples.reset_index(drop=True)

    return samples
