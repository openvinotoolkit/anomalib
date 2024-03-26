"""Base Video Dataset."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch
from pandas import DataFrame
from torchvision.transforms.v2 import Transform
from torchvision.transforms.v2.functional import to_dtype_video
from torchvision.tv_tensors import Mask

from anomalib import TaskType
from anomalib.data.base.datamodule import AnomalibDataModule
from anomalib.data.base.dataset import AnomalibDataset
from anomalib.data.utils import ValSplitMode, masks_to_boxes
from anomalib.data.utils.video import ClipsIndexer

if TYPE_CHECKING:
    from collections.abc import Callable


class VideoTargetFrame(str, Enum):
    """Target frame for a video-clip.

    Used in multi-frame models to determine which frame's ground truth information will be used.
    """

    FIRST = "first"
    LAST = "last"
    MID = "mid"
    ALL = "all"


class AnomalibVideoDataset(AnomalibDataset, ABC):
    """Base video anomalib dataset class.

    Args:
        task (str): Task type, either 'classification' or 'segmentation'
        clip_length_in_frames (int): Number of video frames in each clip.
        frames_between_clips (int): Number of frames between each consecutive video clip.
        transform (Transform, optional): Transforms that should be applied to the input clips.
            Defaults to ``None``.
        target_frame (VideoTargetFrame): Specifies the target frame in the video clip, used for ground truth retrieval.
            Defaults to ``VideoTargetFrame.LAST``.
    """

    def __init__(
        self,
        task: TaskType,
        clip_length_in_frames: int,
        frames_between_clips: int,
        transform: Transform | None = None,
        target_frame: VideoTargetFrame = VideoTargetFrame.LAST,
    ) -> None:
        super().__init__(task, transform)

        self.clip_length_in_frames = clip_length_in_frames
        self.frames_between_clips = frames_between_clips
        self.transform = transform

        self.indexer: ClipsIndexer | None = None
        self.indexer_cls: Callable | None = None

        self.target_frame = target_frame

    def __len__(self) -> int:
        """Get length of the dataset."""
        if not isinstance(self.indexer, ClipsIndexer):
            msg = "self.indexer must be an instance of ClipsIndexer."
            raise TypeError(msg)
        return self.indexer.num_clips()

    @property
    def samples(self) -> DataFrame:
        """Get the samples dataframe."""
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

        Should be called after each change to self._samples
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

    def _select_targets(self, item: dict[str, Any]) -> dict[str, Any]:
        """Select the target frame from the clip.

        Args:
            item (dict[str, Any]): Item containing the clip information.

        Raises:
            ValueError: If the target frame is not one of the supported options.

        Returns:
            dict[str, Any]: Selected item from the clip.
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

        if item.get("mask") is not None:
            item["mask"] = item["mask"][idx, ...]
        if item.get("boxes") is not None:
            item["boxes"] = item["boxes"][idx]
        if item.get("label") is not None:
            item["label"] = item["label"][idx]
        if item.get("original_image") is not None:
            item["original_image"] = item["original_image"][idx]
        if item.get("frames") is not None:
            item["frames"] = item["frames"][idx]
        return item

    def __getitem__(self, index: int) -> dict[str, str | torch.Tensor]:
        """Get the dataset item for the index ``index``.

        Args:
            index (int): Index of the item to be returned.

        Returns:
            dict[str, str | torch.Tensor]: Dictionary containing the mask, clip and file system information.
        """
        if not isinstance(self.indexer, ClipsIndexer):
            msg = "self.indexer must be an instance of ClipsIndexer."
            raise TypeError(msg)
        item = self.indexer.get_item(index)
        item["image"] = to_dtype_video(video=item["image"], scale=True)
        # include the untransformed image for visualization
        item["original_image"] = item["image"].to(torch.uint8)

        # apply transforms
        if item.get("mask") is not None:
            if self.transform:
                item["image"], item["mask"] = self.transform(item["image"], Mask(item["mask"]))
            item["label"] = torch.Tensor([1 in frame for frame in item["mask"]]).int().squeeze(0)
            if self.task == TaskType.DETECTION:
                item["boxes"], _ = masks_to_boxes(item["mask"])
                item["boxes"] = item["boxes"][0] if len(item["boxes"]) == 1 else item["boxes"]
        elif self.transform:
            item["image"] = self.transform(item["image"])

        # squeeze temporal dimensions in case clip length is 1
        item["image"] = item["image"].squeeze(0)

        # include only target frame in gt
        if self.clip_length_in_frames > 1 and self.target_frame != VideoTargetFrame.ALL:
            item = self._select_targets(item)

        if item["mask"] is None:
            item.pop("mask")

        return item


class AnomalibVideoDataModule(AnomalibDataModule):
    """Base class for video data modules."""

    def _create_test_split(self) -> None:
        """Video datamodules do not support dynamic assignment of the test split."""

    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets and perform dynamic subset splitting.

        This method may be overridden in subclass for custom splitting behaviour.

        Video datamodules are not compatible with synthetic anomaly generation.
        """
        if self.train_data is None:
            msg = "self.train_data cannot be None."
            raise ValueError(msg)

        if self.test_data is None:
            msg = "self.test_data cannot be None."
            raise ValueError(msg)

        self.train_data.setup()
        self.test_data.setup()

        if self.val_split_mode == ValSplitMode.SYNTHETIC:
            msg = f"Val split mode {self.test_split_mode} not supported for video datasets."
            raise ValueError(msg)

        self._create_val_split()
