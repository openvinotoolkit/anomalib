"""UCSD Pedestrian Dataset.

This module provides PyTorch Dataset implementation for the UCSD Pedestrian
dataset for abnormal event detection. The dataset contains surveillance videos
with both normal and abnormal events.

The dataset expects the following directory structure::

    root/
    ├── UCSDped1/
    │   ├── Train/
    │   │   ├── Train001/
    │   │   │   ├── 001.tif
    │   │   │   └── ...
    │   │   └── ...
    │   └── Test/
    │       ├── Test001/
    │       │   ├── 001.tif
    │       │   └── ...
    │       ├── Test001_gt/
    │       │   ├── 001.bmp
    │       │   └── ...
    │       └── ...
    └── UCSDped2/
        ├── Train/
        └── Test/

Example:
    Create a dataset for training:

    >>> from anomalib.data.datasets import UCSDpedDataset
    >>> from anomalib.data.utils import Split
    >>> dataset = UCSDpedDataset(
    ...     root="./datasets/ucsdped",
    ...     category="UCSDped1",
    ...     split=Split.TRAIN
    ... )
    >>> dataset[0].keys()
    dict_keys(['image', 'video_path', 'frames', 'last_frame', 'original_image'])

    Create a test dataset:

    >>> dataset = UCSDpedDataset(
    ...     root="./datasets/ucsdped",
    ...     category="UCSDped1",
    ...     split=Split.TEST
    ... )
    >>> dataset[0].keys()
    dict_keys(['image', 'mask', 'video_path', 'frames', 'last_frame',
    'original_image', 'label'])

License:
    UCSD Pedestrian Dataset is released under the BSD 2-Clause License.

Reference:
    Mahadevan, V., Li, W., Bhalodia, V., & Vasconcelos, N. (2010). Anomaly
    detection in crowded scenes. In IEEE Conference on Computer Vision and
    Pattern Recognition (CVPR), 2010.
"""

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
        root (Path | str): Path to the root of the dataset.
        category (str): Sub-category of the dataset, must be one of ``CATEGORIES``.
        split (str | Split | None): Dataset split - usually ``Split.TRAIN`` or
            ``Split.TEST``.
        clip_length_in_frames (int, optional): Number of video frames in each clip.
            Defaults to ``2``.
        frames_between_clips (int, optional): Number of frames between each
            consecutive video clip. Defaults to ``10``.
        target_frame (VideoTargetFrame): Specifies the target frame in the video
            clip, used for ground truth retrieval. Defaults to
            ``VideoTargetFrame.LAST``.
        augmentations (Transform, optional): Augmentations that should be applied to the input images.
            Defaults to ``None``.

    Example:
        >>> from pathlib import Path
        >>> from anomalib.data.datasets import UCSDpedDataset
        >>> dataset = UCSDpedDataset(
        ...     root=Path("./datasets/ucsdped"),
        ...     category="UCSDped1",
        ...     split="train"
        ... )
        >>> dataset[0].keys()
        dict_keys(['image', 'video_path', 'frames', 'last_frame',
        'original_image'])
    """

    def __init__(
        self,
        root: str | Path,
        category: str,
        split: Split,
        clip_length_in_frames: int = 2,
        frames_between_clips: int = 10,
        target_frame: VideoTargetFrame = VideoTargetFrame.LAST,
        augmentations: Transform | None = None,
    ) -> None:
        super().__init__(
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            target_frame=target_frame,
            augmentations=augmentations,
        )

        self.root_category = Path(root) / category
        self.split = split
        self.indexer_cls: Callable = UCSDpedClipsIndexer
        self.samples = make_ucsd_dataset(self.root_category, self.split)


class UCSDpedClipsIndexer(ClipsIndexer):
    """Clips class for UCSDped dataset."""

    def get_mask(self, idx: int) -> np.ndarray | None:
        """Retrieve the masks from the file system.

        Args:
            idx (int): Index of the clip.

        Returns:
            np.ndarray | None: Stack of mask frames if available, None otherwise.
        """
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
            idx (int): Index of the subclip. Must be between 0 and num_clips().

        Returns:
            tuple[torch.Tensor, torch.Tensor, dict[str, Any], int]: Tuple
            containing:
                - video frames tensor
                - empty audio tensor
                - empty info dict
                - video index

        Raises:
            IndexError: If ``idx`` is out of range.
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

    The files are expected to follow the structure::

        path/to/dataset/category/split/video_id/image_filename.tif
        path/to/dataset/category/split/video_id_gt/mask_filename.bmp

    Args:
        path (Path): Path to dataset.
        split (str | Split | None, optional): Dataset split (ie., either train or
            test). Defaults to ``None``.

    Example:
        The following example shows how to get testing samples from UCSDped2
        category:

        >>> root = Path('./UCSDped')
        >>> category = 'UCSDped2'
        >>> path = root / category
        >>> path
        PosixPath('UCSDped/UCSDped2')

        >>> samples = make_ucsd_dataset(path, split='test')
        >>> samples.head()
           root             folder image_path                    mask_path
        0  UCSDped/UCSDped2 Test   UCSDped/UCSDped2/Test/Test001 UCSDped/...

    Returns:
        DataFrame: Output dataframe containing samples for the requested split.
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
