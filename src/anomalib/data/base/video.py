"""Base Video Dataset."""

from __future__ import annotations

from abc import ABC
from enum import Enum
from typing import Callable

import albumentations as A
import torch
from pandas import DataFrame
from torch import Tensor

from anomalib.data.base.datamodule import AnomalibDataModule
from anomalib.data.base.dataset import AnomalibDataset
from anomalib.data.task_type import TaskType
from anomalib.data.utils import ValSplitMode, masks_to_boxes
from anomalib.data.utils.video import ClipsIndexer


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
        transform (A.Compose): Albumentations Compose object describing the transforms that are applied to the inputs.
        clip_length_in_frames (int): Number of video frames in each clip.
        frames_between_clips (int): Number of frames between each consecutive video clip.
        target_frame (VideoTargetFrame): Specifies the target frame in the video clip, used for ground truth retrieval
    """

    def __init__(
        self,
        task: TaskType,
        transform: A.Compose,
        clip_length_in_frames: int,
        frames_between_clips: int,
        target_frame=VideoTargetFrame.LAST,
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
        assert isinstance(self.indexer, ClipsIndexer)
        return self.indexer.num_clips()

    @property
    def samples(self) -> DataFrame:
        """Get the samples dataframe."""
        return super().samples

    @samples.setter
    def samples(self, samples):
        """Overwrite samples and re-index subvideos."""
        super(AnomalibVideoDataset, self.__class__).samples.fset(self, samples)
        self._setup_clips()

    def _setup_clips(self) -> None:
        """Compute the video and frame indices of the subvideos.

        Should be called after each change to self._samples
        """
        assert callable(self.indexer_cls)
        self.indexer = self.indexer_cls(  # pylint: disable=not-callable
            video_paths=list(self.samples.image_path),
            mask_paths=list(self.samples.mask_path),
            clip_length_in_frames=self.clip_length_in_frames,
            frames_between_clips=self.frames_between_clips,
        )

    def _select_targets(self, item):
        if self.target_frame == VideoTargetFrame.FIRST:
            idx = 0
        elif self.target_frame == VideoTargetFrame.LAST:
            idx = -1
        elif self.target_frame == VideoTargetFrame.MID:
            idx = int(self.clip_length_in_frames / 2)
        else:
            raise ValueError(f"Unknown video target frame: {self.target_frame}")

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

    def __getitem__(self, index: int) -> dict[str, str | Tensor]:
        """Return mask, clip and file system information."""
        assert isinstance(self.indexer, ClipsIndexer)

        item = self.indexer.get_item(index)
        # include the untransformed image for visualization
        item["original_image"] = item["image"].to(torch.uint8)

        # apply transforms
        if "mask" in item and item["mask"] is not None:
            processed_frames = [
                self.transform(image=frame.numpy(), mask=mask) for frame, mask in zip(item["image"], item["mask"])
            ]
            item["image"] = torch.stack([item["image"] for item in processed_frames]).squeeze(0)
            mask = torch.as_tensor(item["mask"])
            item["mask"] = torch.stack([item["mask"] for item in processed_frames]).squeeze(0)
            item["label"] = Tensor([1 in frame for frame in mask]).int().squeeze(0)
            if self.task == TaskType.DETECTION:
                item["boxes"], _ = masks_to_boxes(item["mask"])
                item["boxes"] = item["boxes"][0] if len(item["boxes"]) == 1 else item["boxes"]
        else:
            item["image"] = torch.stack(
                [self.transform(image=frame.numpy())["image"] for frame in item["image"]]
            ).squeeze(0)

        # include only target frame in gt
        if self.clip_length_in_frames > 1 and self.target_frame != VideoTargetFrame.ALL:
            item = self._select_targets(item)

        if item["mask"] is None:
            item.pop("mask")

        return item


class AnomalibVideoDataModule(AnomalibDataModule):
    """Base class for video data modules."""

    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets and perform dynamic subset splitting.

        This method may be overridden in subclass for custom splitting behaviour.

        Video datamodules are not compatible with synthetic anomaly generation.
        """
        assert self.train_data is not None
        assert self.test_data is not None

        self.train_data.setup()
        self.test_data.setup()

        if self.val_split_mode == ValSplitMode.SYNTHETIC:
            raise ValueError(f"Val split mode {self.test_split_mode} not supported for video datasets.")

        self._create_val_split()
