"""CUHK Avenue Dataset.

Description:
    This script contains PyTorch Dataset for the CUHK Avenue dataset.
    If the dataset is not already present on the file system, the DataModule class will download and
    extract the dataset, converting the .mat mask files to .png format.

Reference:
    - Lu, Cewu, Jianping Shi, and Jiaya Jia. "Abnormal event detection at 150 fps in Matlab."
      In Proceedings of the IEEE International Conference on Computer Vision, 2013.
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
    """Avenue Dataset class.

    Args:
        split (Split): Split of the dataset, usually Split.TRAIN or Split.TEST
        root (Path | str): Path to the root of the dataset
            Defaults to ``./datasets/avenue``.
        gt_dir (Path | str): Path to the ground truth files
            Defaults to ``./datasets/avenue/ground_truth_demo``.
        clip_length_in_frames (int, optional): Number of video frames in each clip.
            Defaults to ``2``.
        frames_between_clips (int, optional): Number of frames between each consecutive video clip.
            Defaults to ``1``.
        target_frame (VideoTargetFrame): Specifies the target frame in the video clip, used for ground truth retrieval.
            Defaults to ``VideoTargetFrame.LAST``.
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.

    Examples:
        To create an Avenue dataset to train a model:

        .. code-block:: python

            dataset = AvenueDataset(
                transform=transform,
                split="test",
                root="./datasets/avenue/",
            )

            dataset.setup()
            dataset[0].keys()

            # Output: dict_keys(['image', 'mask', 'video_path', 'frames', 'last_frame', 'original_image', 'label'])

        Avenue video dataset can also be used as an image dataset if you set the clip length to 1. This means that each
        video frame will be treated as a separate sample. This is useful for training an image model on the
        Avenue dataset. The following code shows how to create an image dataset:

        .. code-block:: python

            dataset = AvenueDataset(
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
        split: Split,
        root: Path | str = "./datasets/avenue",
        gt_dir: Path | str = "./datasets/avenue/ground_truth_demo",
        clip_length_in_frames: int = 2,
        frames_between_clips: int = 1,
        transform: Transform | None = None,
        target_frame: VideoTargetFrame = VideoTargetFrame.LAST,
    ) -> None:
        super().__init__(
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            target_frame=target_frame,
            transform=transform,
        )

        self.root = root if isinstance(root, Path) else Path(root)
        self.gt_dir = gt_dir if isinstance(gt_dir, Path) else Path(gt_dir)
        self.split = split
        self.indexer_cls: Callable = AvenueClipsIndexer
        self.samples = make_avenue_dataset(self.root, self.gt_dir, self.split)


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
            masks = torch.stack([read_mask(mask_path, as_tensor=True) for mask_path in mask_paths])
        else:
            mat = scipy.io.loadmat(matfile)
            masks = np.vstack([np.stack(m) for m in mat["volLabel"]])
            masks = np.take(masks, frames, 0)
        return masks
