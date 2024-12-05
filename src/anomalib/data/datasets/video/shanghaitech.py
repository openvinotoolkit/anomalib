"""ShanghaiTech Campus Dataset.

Description:
    This script contains PyTorch Dataset for the ShanghaiTech Campus dataset.
    If the dataset is not on the file system, the DataModule class downloads and
    extracts the dataset and converts video files to a format that is readable by pyav.

License:
    ShanghaiTech Campus Dataset is released under the BSD 2-Clause License.

Reference:
    - W. Liu and W. Luo, D. Lian and S. Gao. "Future Frame Prediction for Anomaly Detection -- A New Baseline."
      IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2018.
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
        split (Split): Split of the dataset, usually Split.TRAIN or Split.TEST
        root (Path | str): Path to the root of the dataset
        scene (int): Index of the dataset scene (category) in range [1, 13]
        clip_length_in_frames (int, optional): Number of video frames in each clip.
        frames_between_clips (int, optional): Number of frames between each consecutive video clip.
        target_frame (VideoTargetFrame): Specifies the target frame in the video clip, used for ground truth retrieval.
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
    """

    def __init__(
        self,
        split: Split,
        root: Path | str = "./datasets/shanghaitech",
        scene: int = 1,
        clip_length_in_frames: int = 2,
        frames_between_clips: int = 1,
        target_frame: VideoTargetFrame = VideoTargetFrame.LAST,
        transform: Transform | None = None,
    ) -> None:
        super().__init__(
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            target_frame=target_frame,
            transform=transform,
        )

        self.root = Path(root)
        self.scene = scene
        self.split = split
        self.indexer_cls = ShanghaiTechTrainClipsIndexer if self.split == Split.TRAIN else ShanghaiTechTestClipsIndexer
        self.samples = make_shanghaitech_dataset(self.root, self.scene, self.split)


class ShanghaiTechTrainClipsIndexer(ClipsIndexer):
    """Clips indexer for ShanghaiTech dataset.

    The train and test subsets of the ShanghaiTech dataset use different file formats, so separate
    clips indexer implementations are needed.
    """

    @staticmethod
    def get_mask(idx: int) -> torch.Tensor | None:
        """No masks available for training set."""
        del idx  # Unused argument
        return None


class ShanghaiTechTestClipsIndexer(ClipsIndexer):
    """Clips indexer for the test set of the ShanghaiTech Campus dataset.

    The train and test subsets of the ShanghaiTech dataset use different file formats, so separate
    clips indexer implementations are needed.
    """

    def get_mask(self, idx: int) -> torch.Tensor | None:
        """Retrieve the masks from the file system."""
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

        self.video_fps = [None] * len(self.video_paths)  # fps information cannot be inferred from folder structure

    def get_clip(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], int]:
        """Get a subclip from a list of videos.

        Args:
            idx (int): index of the subclip. Must be between 0 and num_clips().

        Returns:
            video (torch.Tensor)
            audio (torch.Tensor)
            info (Dict)
            video_idx (int): index of the video in `video_paths`
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
