"""Torch-based dataclasses for video data in Anomalib.

This module provides PyTorch-based implementations of the generic dataclasses
used in Anomalib for video data. These classes are designed to work with PyTorch
tensors for efficient data handling and processing in anomaly detection tasks.

The module contains two main classes:
    - :class:`VideoItem`: For single video data items
    - :class:`VideoBatch`: For batched video data items

Example:
    Create and use a torch video item::

        >>> from anomalib.data.dataclasses.torch import VideoItem
        >>> import torch
        >>> item = VideoItem(
        ...     image=torch.rand(10, 3, 224, 224),  # 10 frames
        ...     gt_label=torch.tensor(0),
        ...     video_path="path/to/video.mp4"
        ... )
        >>> item.image.shape
        torch.Size([10, 3, 224, 224])
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
    VideoValidator,
    _VideoInputFields[torch.Tensor, Video, Mask, str],
    DatasetItem[Video],
):
    """Dataclass for individual video items in Anomalib datasets using PyTorch.

    This class combines :class:`_VideoInputFields` and :class:`DatasetItem` for
    video-based anomaly detection. It includes video-specific fields and
    validation methods to ensure proper formatting for Anomalib's video-based
    models.

    The class uses the following type parameters:
        - Video: :class:`torch.Tensor` with shape ``(T, C, H, W)``
        - Label: :class:`torch.Tensor`
        - Mask: :class:`torch.Tensor` with shape ``(T, H, W)``
        - Path: :class:`str`

    Where ``T`` represents the temporal dimension (number of frames).

    Example:
        >>> import torch
        >>> from anomalib.data.dataclasses.torch import VideoItem
        >>> item = VideoItem(
        ...     image=torch.rand(10, 3, 224, 224),  # 10 frames
        ...     gt_label=torch.tensor(0),
        ...     video_path="path/to/video.mp4"
        ... )
        >>> item.image.shape
        torch.Size([10, 3, 224, 224])

        Convert to numpy format:
        >>> numpy_item = item.to_numpy()
        >>> type(numpy_item).__name__
        'NumpyVideoItem'
    """

    numpy_class = NumpyVideoItem

    def to_image(self) -> ImageItem:
        """Convert the video item to an image item."""
        image_keys = [field.name for field in fields(ImageItem)]
        return ImageItem(**{key: getattr(self, key, None) for key in image_keys})


@dataclass
class VideoBatch(
    ToNumpyMixin[NumpyVideoBatch],
    BatchIterateMixin[VideoItem],
    VideoBatchValidator,
    _VideoInputFields[torch.Tensor, Video, Mask, list[str]],
    Batch[Video],
):
    """Dataclass for batches of video items in Anomalib datasets using PyTorch.

    This class represents batches of video data for batch processing in anomaly
    detection tasks. It combines functionality from multiple mixins to handle
    batched video data efficiently.

    The class uses the following type parameters:
        - Video: :class:`torch.Tensor` with shape ``(B, T, C, H, W)``
        - Label: :class:`torch.Tensor` with shape ``(B,)``
        - Mask: :class:`torch.Tensor` with shape ``(B, T, H, W)``
        - Path: :class:`list` of :class:`str`

    Where ``B`` represents the batch dimension and ``T`` the temporal dimension.

    Example:
        >>> import torch
        >>> from anomalib.data.dataclasses.torch import VideoBatch
        >>> batch = VideoBatch(
        ...     image=torch.rand(32, 10, 3, 224, 224),  # 32 videos, 10 frames
        ...     gt_label=torch.randint(0, 2, (32,)),
        ...     video_path=["video_{}.mp4".format(i) for i in range(32)]
        ... )
        >>> batch.image.shape
        torch.Size([32, 10, 3, 224, 224])

        Iterate over items in batch:
        >>> next(iter(batch)).image.shape
        torch.Size([10, 3, 224, 224])

        Convert to numpy format:
        >>> numpy_batch = batch.to_numpy()
        >>> type(numpy_batch).__name__
        'NumpyVideoBatch'
    """

    item_class = VideoItem
    numpy_class = NumpyVideoBatch
