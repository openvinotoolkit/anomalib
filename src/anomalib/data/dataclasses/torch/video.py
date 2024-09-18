"""Torch-based dataclasses for video data in Anomalib.

This module provides PyTorch-based implementations of the generic dataclasses
used in Anomalib for video data. These classes are designed to work with PyTorch
tensors for efficient data handling and processing in anomaly detection tasks.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, fields

import torch
from torchvision.tv_tensors import Image, Mask, Video

from anomalib.data.dataclasses.generic import BatchIterateMixin, _VideoInputFields
from anomalib.data.dataclasses.numpy.video import NumpyVideoBatch, NumpyVideoItem
from anomalib.data.dataclasses.torch.base import Batch, DatasetItem, ToNumpyMixin
from anomalib.data.dataclasses.torch.image import ImageItem


@dataclass
class VideoItem(
    ToNumpyMixin[NumpyVideoItem],
    _VideoInputFields[torch.Tensor, Video, Mask, str],
    DatasetItem[Video],
):
    """Dataclass for individual video items in Anomalib datasets using PyTorch tensors.

    This class represents a single video item in Anomalib datasets using PyTorch tensors.
    It combines the functionality of ToNumpyMixin, _VideoInputFields, and DatasetItem
    to handle video data, including frames, labels, masks, and metadata.

    Examples:
        >>> item = VideoItem(
        ...     image=torch.rand(10, 3, 224, 224),  # 10 frames
        ...     gt_label=torch.tensor(1),
        ...     gt_mask=torch.rand(10, 224, 224) > 0.5,
        ...     video_path="path/to/video.mp4"
        ... )

        >>> print(item.image.shape)
        torch.Size([10, 3, 224, 224])

        >>> numpy_item = item.to_numpy()
        >>> print(type(numpy_item))
        <class 'anomalib.dataclasses.numpy.NumpyVideoItem'>
    """

    numpy_class = NumpyVideoItem

    @staticmethod
    def _validate_image(image: Image) -> Video:
        return image

    @staticmethod
    def _validate_gt_label(gt_label: torch.Tensor) -> torch.Tensor:
        return gt_label

    @staticmethod
    def _validate_gt_mask(gt_mask: Mask) -> Mask:
        return gt_mask

    @staticmethod
    def _validate_mask_path(mask_path: str) -> str:
        return mask_path

    @staticmethod
    def _validate_anomaly_map(anomaly_map: torch.Tensor) -> torch.Tensor | None:
        return anomaly_map

    @staticmethod
    def _validate_pred_score(pred_score: torch.Tensor | None) -> torch.Tensor | None:
        return pred_score

    @staticmethod
    def _validate_pred_mask(pred_mask: torch.Tensor) -> torch.Tensor | None:
        return pred_mask

    @staticmethod
    def _validate_pred_label(pred_label: torch.Tensor) -> torch.Tensor | None:
        return pred_label

    @staticmethod
    def _validate_original_image(original_image: Video) -> Video:
        return original_image

    @staticmethod
    def _validate_video_path(video_path: str) -> str:
        return video_path

    @staticmethod
    def _validate_target_frame(target_frame: torch.Tensor) -> torch.Tensor:
        return target_frame

    @staticmethod
    def _validate_frames(frames: torch.Tensor) -> torch.Tensor:
        return frames

    @staticmethod
    def _validate_last_frame(last_frame: torch.Tensor) -> torch.Tensor:
        return last_frame

    def to_image(self) -> ImageItem:
        """Convert the video item to an image item."""
        image_keys = [field.name for field in fields(ImageItem)]
        return ImageItem(**{key: getattr(self, key, None) for key in image_keys})


@dataclass
class VideoBatch(
    ToNumpyMixin[NumpyVideoBatch],
    BatchIterateMixin[VideoItem],
    _VideoInputFields[torch.Tensor, Video, Mask, list[str]],
    Batch[Video],
):
    """Dataclass for batches of video items in Anomalib datasets using PyTorch tensors.

    This class represents a batch of video items in Anomalib datasets using PyTorch tensors.
    It combines the functionality of ToNumpyMixin, BatchIterateMixin, _VideoInputFields,
    and Batch to handle batches of video data, including frames, labels, masks, and metadata.

    Examples:
        >>> batch = VideoBatch(
        ...     image=torch.rand(32, 10, 3, 224, 224),  # 32 videos, 10 frames each
        ...     gt_label=torch.randint(0, 2, (32,)),
        ...     gt_mask=torch.rand(32, 10, 224, 224) > 0.5,
        ...     video_path=["path/to/video_{}.mp4".format(i) for i in range(32)]
        ... )

        >>> print(batch.image.shape)
        torch.Size([32, 10, 3, 224, 224])

        >>> for item in batch:
        ...     print(item.image.shape)
        torch.Size([10, 3, 224, 224])

        >>> numpy_batch = batch.to_numpy()
        >>> print(type(numpy_batch))
        <class 'anomalib.dataclasses.numpy.NumpyVideoBatch'>
    """

    item_class = VideoItem
    numpy_class = NumpyVideoBatch

    @staticmethod
    def _validate_image(image: Image) -> Video:
        return image

    @staticmethod
    def _validate_gt_label(gt_label: torch.Tensor) -> torch.Tensor:
        return gt_label

    @staticmethod
    def _validate_gt_mask(gt_mask: Mask) -> Mask:
        return gt_mask

    @staticmethod
    def _validate_mask_path(mask_path: list[str]) -> list[str]:
        return mask_path

    @staticmethod
    def _validate_anomaly_map(anomaly_map: torch.Tensor) -> torch.Tensor:
        return anomaly_map

    @staticmethod
    def _validate_pred_score(pred_score: torch.Tensor) -> torch.Tensor:
        return pred_score

    @staticmethod
    def _validate_pred_mask(pred_mask: torch.Tensor) -> torch.Tensor:
        return pred_mask

    @staticmethod
    def _validate_pred_label(pred_label: torch.Tensor) -> torch.Tensor:
        return pred_label

    @staticmethod
    def _validate_original_image(original_image: Video) -> Video:
        return original_image

    @staticmethod
    def _validate_video_path(video_path: list[str]) -> list[str]:
        return video_path

    @staticmethod
    def _validate_target_frame(target_frame: torch.Tensor) -> torch.Tensor:
        return target_frame

    @staticmethod
    def _validate_frames(frames: torch.Tensor) -> torch.Tensor:
        return frames

    @staticmethod
    def _validate_last_frame(last_frame: torch.Tensor) -> torch.Tensor:
        return last_frame
