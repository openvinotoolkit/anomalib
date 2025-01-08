"""CUHK Avenue Dataset.

This module provides PyTorch Dataset implementation for the CUHK Avenue dataset
for abnormal event detection. The dataset contains surveillance videos with both
normal and abnormal events.

If the dataset is not already present on the file system, the DataModule class
will download and extract the dataset, converting the .mat mask files to .png
format.

Example:
    Create a dataset for training:

    >>> from anomalib.data.datasets import AvenueDataset
    >>> dataset = AvenueDataset(
    ...     root="./datasets/avenue",
    ...     split="train"
    ... )
    >>> dataset.setup()
    >>> dataset[0].keys()
    dict_keys(['image', 'mask', 'video_path', 'frames', 'last_frame',
    'original_image', 'label'])

    Create an image dataset by setting ``clip_length_in_frames=1``:

    >>> dataset = AvenueDataset(
    ...     root="./datasets/avenue",
    ...     split="test",
    ...     clip_length_in_frames=1
    ... )
    >>> dataset.setup()
    >>> dataset[0]["image"].shape
    torch.Size([3, 256, 256])

Reference:
    Lu, Cewu, Jianping Shi, and Jiaya Jia. "Abnormal event detection at 150 fps
    in Matlab." In Proceedings of the IEEE International Conference on Computer
    Vision, 2013.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy
import torch
from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base.video import AnomalibVideoDataset, VideoTargetFrame
from anomalib.data.utils import Split, read_mask, validate_path
from anomalib.data.utils.video import ClipsIndexer

if TYPE_CHECKING:
    from collections.abc import Callable


class AvenueDataset(AnomalibVideoDataset):
    """CUHK Avenue dataset class.

    Args:
        split (Split): Dataset split - usually ``Split.TRAIN`` or ``Split.TEST``
        root (Path | str, optional): Path to the root directory containing the
            dataset. Defaults to ``"./datasets/avenue"``.
        gt_dir (Path | str, optional): Path to the ground truth directory.
            Defaults to ``"./datasets/avenue/ground_truth_demo"``.
        clip_length_in_frames (int, optional): Number of frames in each video
            clip. Defaults to ``2``.
        frames_between_clips (int, optional): Number of frames between
            consecutive video clips. Defaults to ``1``.
        target_frame (VideoTargetFrame, optional): Target frame in the video
            clip for ground truth retrieval. Defaults to
            ``VideoTargetFrame.LAST``.
        augmentations (Transform, optional): Augmentations that should be applied to the input images.
            Defaults to ``None``.

    Example:
        Create a dataset for testing:

        >>> dataset = AvenueDataset(
        ...     root="./datasets/avenue",
        ...     split="test",
        ...     transform=transform
        ... )
        >>> dataset.setup()
        >>> dataset[0].keys()
        dict_keys(['image', 'mask', 'video_path', 'frames', 'last_frame',
        'original_image', 'label'])
    """

    def __init__(
        self,
        split: Split,
        root: Path | str = "./datasets/avenue",
        gt_dir: Path | str = "./datasets/avenue/ground_truth_demo",
        clip_length_in_frames: int = 2,
        frames_between_clips: int = 1,
        augmentations: Transform | None = None,
        target_frame: VideoTargetFrame = VideoTargetFrame.LAST,
    ) -> None:
        super().__init__(
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            target_frame=target_frame,
            augmentations=augmentations,
        )

        self.root = root if isinstance(root, Path) else Path(root)
        self.gt_dir = gt_dir if isinstance(gt_dir, Path) else Path(gt_dir)
        self.split = split
        self.indexer_cls: Callable = AvenueClipsIndexer
        self.samples = make_avenue_dataset(self.root, self.gt_dir, self.split)


def make_avenue_dataset(
    root: Path,
    gt_dir: Path,
    split: Split | str | None = None,
) -> DataFrame:
    """Create CUHK Avenue dataset by parsing the file structure.

    The files are expected to follow the structure:
        path/to/dataset/[training_videos|testing_videos]/video_filename.avi
        path/to/ground_truth/mask_filename.mat

    Args:
        root (Path): Path to dataset root directory
        gt_dir (Path): Path to ground truth directory
        split (Split | str | None, optional): Dataset split (train/test).
            Defaults to ``None``.

    Example:
        Get testing samples from Avenue dataset:

        >>> root = Path("./avenue")
        >>> gt_dir = Path("./avenue/masks")
        >>> samples = make_avenue_dataset(root, gt_dir, split="test")
        >>> samples.head()
           root     folder    image_path    mask_path   split
        0  ./avenue testing  01.avi        01_label.mat test
        1  ./avenue testing  02.avi        02_label.mat test

    Returns:
        DataFrame: Dataframe containing samples for the requested split
    """
    root = validate_path(root)

    samples_list = [(str(root),) + filename.parts[-2:] for filename in root.glob("**/*.avi")]
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

    # infer the task type
    samples.attrs["task"] = "classification" if (samples["mask_path"] == "").all() else "segmentation"

    if split:
        samples = samples[samples.split == split]
        samples = samples.reset_index(drop=True)

    return samples


class AvenueClipsIndexer(ClipsIndexer):
    """Clips indexer class for Avenue dataset.

    This class handles retrieving video clips and corresponding masks from the
    Avenue dataset.
    """

    def get_mask(self, idx: int) -> np.ndarray | None:
        """Retrieve masks from the file system.

        Args:
            idx (int): Index of the clip

        Returns:
            np.ndarray | None: Array of masks if available, else None
        """
        video_idx, frames_idx = self.get_clip_location(idx)
        matfile = self.mask_paths[video_idx]
        if matfile == "":  # no gt masks available for this clip
            return None
        frames = self.clips[video_idx][frames_idx]

        # read masks from .png files if available, otherwise from mat files
        mask_folder = Path(matfile).with_suffix("")
        if mask_folder.exists():
            mask_frames = sorted(mask_folder.glob("*"))
            mask_paths = [mask_frames[idx] for idx in frames.int()]
            masks = torch.stack([read_mask(mask_path, as_tensor=True) for mask_path in mask_paths])
        else:
            mat = scipy.io.loadmat(matfile)
            masks = np.vstack([np.stack(m) for m in mat["volLabel"]])
            masks = np.take(masks, frames, 0)
        return masks
