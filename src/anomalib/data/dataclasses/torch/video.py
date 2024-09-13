"""Torch-based dataclasses for video data in Anomalib.

This module provides PyTorch-based implementations of the generic dataclasses
used in Anomalib for video data. These classes are designed to work with PyTorch
tensors for efficient data handling and processing in anomaly detection tasks.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, fields

import torch
from torchvision.tv_tensors import Mask, Video

from anomalib.data.dataclasses.generic import BatchIterateMixin, _VideoInputFields
from anomalib.data.dataclasses.numpy.video import NumpyVideoBatch, NumpyVideoItem
from anomalib.data.dataclasses.torch.base import Batch, DatasetItem, ToNumpyMixin
from anomalib.data.dataclasses.torch.image import ImageItem
from anomalib.data.validators.torch.video import VideoBatchValidator, VideoValidator


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
    def validate_image(image: Video) -> Video:
        """Validate the image."""
        return VideoValidator.validate_image(image)

    @staticmethod
    def validate_gt_label(gt_label: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the ground truth label."""
        return VideoValidator.validate_gt_label(gt_label)

    @staticmethod
    def validate_gt_mask(gt_mask: Mask | None) -> Mask | None:
        """Validate the ground truth mask."""
        return VideoValidator.validate_gt_mask(gt_mask)

    @staticmethod
    def validate_mask_path(mask_path: str | None) -> str | None:
        """Validate the mask path."""
        return VideoValidator.validate_mask_path(mask_path)

    @staticmethod
    def validate_anomaly_map(anomaly_map: torch.Tensor) -> torch.Tensor | None:
        """Validate the anomaly map."""
        return VideoValidator.validate_anomaly_map(anomaly_map)

    @staticmethod
    def validate_pred_score(pred_score: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the prediction score."""
        return VideoValidator.validate_pred_score(pred_score)

    @staticmethod
    def validate_pred_mask(pred_mask: torch.Tensor) -> torch.Tensor | None:
        """Validate the prediction mask."""
        return VideoValidator.validate_pred_mask(pred_mask)

    @staticmethod
    def validate_pred_label(pred_label: torch.Tensor) -> torch.Tensor | None:
        """Validate the prediction label."""
        return VideoValidator.validate_pred_label(pred_label)

    @staticmethod
    def validate_original_image(original_image: torch.Tensor | Video | None) -> torch.Tensor | Video | None:
        """Validate the original image."""
        return VideoValidator.validate_original_image(original_image)

    @staticmethod
    def validate_video_path(video_path: str | None) -> str | None:
        """Validate the video path."""
        return VideoValidator.validate_video_path(video_path)

    @staticmethod
    def validate_target_frame(target_frame: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the target frame."""
        return VideoValidator.validate_target_frame(target_frame)

    @staticmethod
    def validate_frames(frames: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the frames."""
        return VideoValidator.validate_frames(frames)

    @staticmethod
    def validate_last_frame(last_frame: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the last frame."""
        return VideoValidator.validate_last_frame(last_frame)

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
    def validate_image(image: Video) -> Video:
        """Validate the image."""
        return VideoBatchValidator.validate_image(image)

    @staticmethod
    def validate_gt_label(gt_label: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the ground truth label."""
        return VideoBatchValidator.validate_gt_label(gt_label)

    @staticmethod
    def validate_gt_mask(gt_mask: Mask | None) -> Mask | None:
        """Validate the ground truth mask."""
        return VideoBatchValidator.validate_gt_mask(gt_mask)

    @staticmethod
    def validate_mask_path(mask_path: list[str] | None) -> list[str] | None:
        """Validate the mask path."""
        return VideoBatchValidator.validate_mask_path(mask_path)

    @staticmethod
    def validate_anomaly_map(anomaly_map: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the anomaly map."""
        return VideoBatchValidator.validate_anomaly_map(anomaly_map)

    @staticmethod
    def validate_pred_score(pred_score: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the prediction score."""
        return VideoBatchValidator.validate_pred_score(pred_score)

    @staticmethod
    def validate_pred_mask(pred_mask: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the prediction mask."""
        return VideoBatchValidator.validate_pred_mask(pred_mask)

    @staticmethod
    def validate_pred_label(pred_label: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the prediction label."""
        return VideoBatchValidator.validate_pred_label(pred_label)

    @staticmethod
    def validate_original_image(original_image: torch.Tensor | Video | None) -> torch.Tensor | Video | None:
        """Validate the original image."""
        return VideoBatchValidator.validate_original_image(original_image)

    @staticmethod
    def validate_video_path(video_path: list[str] | None) -> list[str] | None:
        """Validate the video path."""
        return VideoBatchValidator.validate_video_path(video_path)

    @staticmethod
    def validate_target_frame(target_frame: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the target frame."""
        return VideoBatchValidator.validate_target_frame(target_frame)

    @staticmethod
    def validate_frames(frames: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the frames."""
        return VideoBatchValidator.validate_frames(frames)

    @staticmethod
    def validate_last_frame(last_frame: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the last frame."""
        return VideoBatchValidator.validate_last_frame(last_frame)
