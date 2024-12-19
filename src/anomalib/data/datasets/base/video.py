"""Base Torch Video Dataset.

This module implements the base video dataset class for anomaly detection tasks that
use video data. The dataset is designed to work with video clips and supports both
classification and segmentation tasks.

Example:
    >>> from anomalib.data.datasets import AnomalibVideoDataset
    >>> dataset = AnomalibVideoDataset(
    ...     clip_length_in_frames=8,
    ...     frames_between_clips=1,
    ...     transform=None,
    ...     target_frame="last"
    ... )
    >>> item = dataset[0]
    >>> item.image.shape
    torch.Size([C, H, W])
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from collections.abc import Callable
from enum import Enum

import torch
from pandas import DataFrame
from torchvision.transforms.v2 import Transform
from torchvision.transforms.v2.functional import to_dtype, to_dtype_video
from torchvision.tv_tensors import Mask

from anomalib.data.dataclasses import VideoBatch, VideoItem
from anomalib.data.utils.video import ClipsIndexer

from .image import AnomalibDataset


class VideoTargetFrame(str, Enum):
    """Target frame for a video-clip.

    Used in multi-frame models to determine which frame's ground truth information
    will be used.

    Args:
        FIRST: Use the first frame in the clip as target
        LAST: Use the last frame in the clip as target
        MID: Use the middle frame in the clip as target
        ALL: Use all frames in the clip as target
    """

    FIRST = "first"
    LAST = "last"
    MID = "mid"
    ALL = "all"


class AnomalibVideoDataset(AnomalibDataset, ABC):
    """Base video anomalib dataset class.

    This class extends ``AnomalibDataset`` to handle video data for anomaly
    detection tasks. It supports both classification and segmentation tasks.

    Args:
        clip_length_in_frames (int): Number of video frames in each clip.
        frames_between_clips (int): Number of frames between each consecutive
            video clip.
        augmentations (Transform, optional): Augmentations that should be applied to the input clips.
            Defaults to ``None``.
        target_frame (VideoTargetFrame, optional): Specifies the target frame in
            the video clip, used for ground truth retrieval.
            Defaults to ``VideoTargetFrame.LAST``.

    Example:
        >>> from torchvision.transforms.v2 import Resize
        >>> dataset = AnomalibVideoDataset(
        ...     clip_length_in_frames=8,
        ...     frames_between_clips=1,
        ...     transform=Resize((256, 256)),
        ...     target_frame="last"
        ... )
        >>> item = dataset[0]
        >>> item.image.shape
        torch.Size([C, H, W])
    """

    def __init__(
        self,
        clip_length_in_frames: int,
        frames_between_clips: int,
        augmentations: Transform | None = None,
        target_frame: VideoTargetFrame = VideoTargetFrame.LAST,
    ) -> None:
        super().__init__(augmentations=augmentations)

        self.clip_length_in_frames = clip_length_in_frames
        self.frames_between_clips = frames_between_clips
        self.augmentations = augmentations

        self.indexer: ClipsIndexer | None = None
        self.indexer_cls: Callable | None = None

        self.target_frame = target_frame

    def __len__(self) -> int:
        """Get length of the dataset.

        Returns:
            int: Number of clips in the dataset.

        Raises:
            TypeError: If ``self.indexer`` is not an instance of ``ClipsIndexer``.
        """
        if not isinstance(self.indexer, ClipsIndexer):
            msg = "self.indexer must be an instance of ClipsIndexer."
            raise TypeError(msg)
        return self.indexer.num_clips()

    @property
    def samples(self) -> DataFrame:
        """Get the samples dataframe.

        Returns:
            DataFrame: DataFrame containing dataset samples.
        """
        return super().samples

    @samples.setter
    def samples(self, samples: DataFrame) -> None:
        """Overwrite samples and re-index subvideos.

        Args:
            samples (DataFrame): DataFrame with new samples.

        Raises:
            ValueError: If the indexer class is not set.
        """
        super(AnomalibVideoDataset, self.__class__).samples.fset(self, samples)  # type: ignore[attr-defined]
        self._setup_clips()

    def _setup_clips(self) -> None:
        """Compute the video and frame indices of the subvideos.

        Should be called after each change to ``self._samples``.

        Raises:
            TypeError: If ``self.indexer_cls`` is not callable.
        """
        if not callable(self.indexer_cls):
            msg = "self.indexer_cls must be callable."
            raise TypeError(msg)
        self.indexer = self.indexer_cls(  # pylint: disable=not-callable
            video_paths=list(self.samples.image_path),
            mask_paths=list(self.samples.mask_path),
            clip_length_in_frames=self.clip_length_in_frames,
            frames_between_clips=self.frames_between_clips,
        )

    def _select_targets(self, item: VideoItem) -> VideoItem:
        """Select the target frame from the clip.

        Args:
            item (VideoItem): Item containing the clip information.

        Returns:
            VideoItem: Selected item from the clip.

        Raises:
            ValueError: If the target frame is not one of the supported options.
        """
        if self.target_frame == VideoTargetFrame.FIRST:
            idx = 0
        elif self.target_frame == VideoTargetFrame.LAST:
            idx = -1
        elif self.target_frame == VideoTargetFrame.MID:
            idx = int(self.clip_length_in_frames / 2)
        else:
            msg = f"Unknown video target frame: {self.target_frame}"
            raise ValueError(msg)

        if item.gt_mask is not None:
            item.gt_mask = item.gt_mask[idx, ...]
        if item.gt_label is not None:
            item.gt_label = item.gt_label[idx]
        if item.original_image is not None:
            item.original_image = item.original_image[idx]
        if item.frames is not None:
            item.frames = item.frames[idx]
        return item

    def __getitem__(self, index: int) -> VideoItem:
        """Get the dataset item for the index.

        Args:
            index (int): Index of the item to be returned.

        Returns:
            VideoItem: Dataset item containing the mask, clip and file system
                information.

        Raises:
            TypeError: If ``self.indexer`` is not an instance of ``ClipsIndexer``.
        """
        if not isinstance(self.indexer, ClipsIndexer):
            msg = "self.indexer must be an instance of ClipsIndexer."
            raise TypeError(msg)
        item = self.indexer.get_item(index)
        item.image = to_dtype_video(video=item.image, scale=True)
        # include the untransformed image for visualization
        item.original_image = to_dtype(item.image, torch.uint8, scale=True)

        # apply transforms
        if item.gt_mask is not None:
            if self.augmentations:
                item.image, item.gt_mask = self.augmentations(item.image, Mask(item.gt_mask))
            item.gt_label = torch.Tensor([1 in frame for frame in item.gt_mask]).int().squeeze(0)
        elif self.augmentations:
            item.image = self.augmentations(item.image)

        # squeeze temporal dimensions in case clip length is 1
        item.image = item.image.squeeze(0)

        # include only target frame in gt
        if self.clip_length_in_frames > 1 and self.target_frame != VideoTargetFrame.ALL:
            item = self._select_targets(item)

        return item

    @property
    def collate_fn(self) -> Callable:
        """Return the collate function for video batches.

        Returns:
            Callable: Collate function for creating video batches.
        """
        return VideoBatch.collate
