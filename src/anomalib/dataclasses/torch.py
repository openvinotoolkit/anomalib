"""Torch-based dataclasses for Anomalib.

This module provides PyTorch-based implementations of the generic dataclasses
used in Anomalib. These classes are designed to work with PyTorch tensors for
efficient data handling and processing in anomaly detection tasks.

These classes extend the generic dataclasses defined in the Anomalib framework,
providing concrete implementations that use PyTorch tensors for tensor-like data.
They include methods for data validation and support operations specific to
image, video, and depth data in the context of anomaly detection.

Note:
    When using these classes, ensure that the input data is in the correct
    format (PyTorch tensors with appropriate shapes) to avoid validation errors.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields
from typing import ClassVar, Generic, NamedTuple, TypeVar

import numpy as np
import torch
from torchvision.transforms.v2.functional import to_dtype_image
from torchvision.tv_tensors import Image, Mask, Video

from .generic import (
    BatchIterateMixin,
    ImageT,
    _DepthInputFields,
    _GenericBatch,
    _GenericItem,
    _ImageInputFields,
    _VideoInputFields,
)
from .numpy import NumpyImageBatch, NumpyImageItem, NumpyVideoBatch, NumpyVideoItem

NumpyT = TypeVar("NumpyT")


class InferenceBatch(NamedTuple):
    """Batch for use in torch and inference models."""

    pred_score: torch.Tensor | None = None
    pred_label: torch.Tensor | None = None
    anomaly_map: torch.Tensor | None = None
    pred_mask: torch.Tensor | None = None


@dataclass
class ToNumpyMixin(Generic[NumpyT]):
    """Mixin for converting torch-based dataclasses to numpy.

    This mixin provides functionality to convert PyTorch tensor data to numpy arrays.
    It requires the subclass to define a 'numpy_class' attribute specifying the
    corresponding numpy-based class.

    Examples:
        >>> from anomalib.dataclasses.numpy import NumpyImageItem
        >>> @dataclass
        ... class TorchImageItem(ToNumpyMixin[NumpyImageItem]):
        ...     numpy_class = NumpyImageItem
        ...     image: torch.Tensor
        ...     gt_label: torch.Tensor

        >>> torch_item = TorchImageItem(image=torch.rand(3, 224, 224), gt_label=torch.tensor(1))
        >>> numpy_item = torch_item.to_numpy()
        >>> isinstance(numpy_item, NumpyImageItem)
        True
    """

    numpy_class: ClassVar[Callable]

    def __init_subclass__(cls, **kwargs) -> None:
        """Ensure that the subclass has the required attributes."""
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "numpy_class"):
            msg = f"{cls.__name__} must have a 'numpy_class' attribute."
            raise AttributeError(msg)

    def to_numpy(self) -> NumpyT:
        """Convert the batch to a NumpyBatch object."""
        batch_dict = asdict(self)
        for key, value in batch_dict.items():
            if isinstance(value, torch.Tensor):
                batch_dict[key] = value.cpu().numpy()
        return self.numpy_class(
            **batch_dict,
        )


@dataclass
class DatasetItem(Generic[ImageT], _GenericItem[torch.Tensor, ImageT, Mask, str]):
    """Base dataclass for individual items in Anomalib datasets using PyTorch tensors.

    This class extends the generic _GenericItem class to provide a PyTorch-specific
    implementation for single data items in Anomalib datasets. It is designed to
    handle various types of data (e.g., images, labels, masks) represented as
    PyTorch tensors.

    The class uses generic types to allow flexibility in the image representation,
    which can vary depending on the specific use case (e.g., standard images, video clips).

    Attributes:
        Inherited from _GenericItem, with PyTorch tensor and Mask types.

    Note:
        This class is typically subclassed to create more specific item types
        (e.g., ImageItem, VideoItem) with additional fields and methods.
    """


@dataclass
class Batch(Generic[ImageT], _GenericBatch[torch.Tensor, ImageT, Mask, list[str]]):
    """Base dataclass for batches of items in Anomalib datasets using PyTorch tensors.

    This class extends the generic _GenericBatch class to provide a PyTorch-specific
    implementation for batches of data in Anomalib datasets. It is designed to
    handle collections of data items (e.g., multiple images, labels, masks)
    represented as PyTorch tensors.

    The class uses generic types to allow flexibility in the image representation,
    which can vary depending on the specific use case (e.g., standard images, video clips).

    Attributes:
        Inherited from _GenericBatch, with PyTorch tensor and Mask types.

    Note:
        This class is typically subclassed to create more specific batch types
        (e.g., ImageBatch, VideoBatch) with additional fields and methods.
    """


@dataclass
class ImageItem(
    ToNumpyMixin[NumpyImageItem],
    _ImageInputFields[str],
    DatasetItem[Image],
):
    """Dataclass for individual image items in Anomalib datasets using PyTorch tensors.

    This class combines the functionality of ToNumpyMixin, _ImageInputFields, and
    DatasetItem to represent single image data points in Anomalib. It includes
    image-specific fields and provides methods for data validation and conversion
    to numpy format.

    The class is designed to work with PyTorch tensors and includes fields for
    the image data, ground truth labels and masks, anomaly maps, and related metadata.

    Attributes:
        Inherited from _ImageInputFields and DatasetItem.

    Methods:
        Inherited from ToNumpyMixin, including to_numpy() for conversion to numpy format.

    Examples:
        >>> item = ImageItem(
        ...     image=torch.rand(3, 224, 224),
        ...     gt_label=torch.tensor(1),
        ...     gt_mask=torch.rand(224, 224) > 0.5,
        ...     image_path="path/to/image.jpg"
        ... )

        >>> print(item.image.shape)
        torch.Size([3, 224, 224])

        >>> numpy_item = item.to_numpy()
        >>> print(type(numpy_item))
        <class 'anomalib.dataclasses.numpy.NumpyImageItem'>
    """

    numpy_class = NumpyImageItem

    def _validate_image(self, image: torch.Tensor) -> Image:
        assert isinstance(image, torch.Tensor), f"Image must be a torch.Tensor, got {type(image)}."
        assert image.ndim == 3, f"Image must have shape [C, H, W], got shape {image.shape}."
        assert image.shape[0] == 3, f"Image must have 3 channels, got {image.shape[0]}."
        return to_dtype_image(image, torch.float32, scale=True)

    def _validate_gt_label(self, gt_label: torch.Tensor | int | None) -> torch.Tensor:
        if gt_label is None:
            return None
        if isinstance(gt_label, int):
            gt_label = torch.tensor(gt_label)
        assert isinstance(
            gt_label,
            torch.Tensor,
        ), f"Ground truth label must be an integer or a torch.Tensor, got {type(gt_label)}."
        assert gt_label.ndim == 0, f"Ground truth label must be a scalar, got shape {gt_label.shape}."
        assert not torch.is_floating_point(gt_label), f"Ground truth label must be boolean or integer, got {gt_label}."
        return gt_label.bool()

    def _validate_gt_mask(self, gt_mask: torch.Tensor | None) -> Mask | None:
        if gt_mask is None:
            return None
        assert isinstance(gt_mask, torch.Tensor), f"Ground truth mask must be a torch.Tensor, got {type(gt_mask)}."
        assert gt_mask.ndim in {
            2,
            3,
        }, f"Ground truth mask must have shape [H, W] or [1, H, W] got shape {gt_mask.shape}."
        if gt_mask.ndim == 3:
            assert gt_mask.shape[0] == 1, f"Ground truth mask must have 1 channel, got {gt_mask.shape[0]}."
            gt_mask = gt_mask.squeeze(0)
        return Mask(gt_mask, dtype=torch.bool)

    def _validate_mask_path(self, mask_path: str | None) -> str | None:
        if mask_path is None:
            return None
        return str(mask_path)

    def _validate_anomaly_map(self, anomaly_map: torch.Tensor | None) -> Mask | None:
        if anomaly_map is None:
            return None
        assert isinstance(anomaly_map, torch.Tensor), f"Anomaly map must be a torch.Tensor, got {type(anomaly_map)}."
        assert anomaly_map.ndim in {
            2,
            3,
        }, f"Anomaly map must have shape [H, W] or [1, H, W], got shape {anomaly_map.shape}."
        if anomaly_map.ndim == 3:
            assert (
                anomaly_map.shape[0] == 1
            ), f"Anomaly map with 3 dimensions must have 1 channel, got {anomaly_map.shape[0]}."
            anomaly_map = anomaly_map.squeeze(0)
        return Mask(anomaly_map, dtype=torch.float32)

    def _validate_pred_score(self, pred_score: torch.Tensor | np.ndarray | None) -> torch.Tensor | None:
        if pred_score is None:
            return torch.amax(self.anomaly_map, dim=(-2, -1)) if self.anomaly_map is not None else None
        if not isinstance(pred_score, torch.Tensor):
            try:
                pred_score = torch.tensor(pred_score)
            except Exception as e:
                msg = "Failed to convert pred_score to a torch.Tensor."
                raise ValueError(msg) from e
        pred_score = pred_score.squeeze()
        assert pred_score.ndim == 0, f"Predicted score must be a scalar, got shape {pred_score.shape}."
        return pred_score.to(torch.float32)

    def _validate_pred_mask(self, pred_mask: torch.Tensor | None) -> Mask | None:
        if pred_mask is None:
            return None
        assert isinstance(pred_mask, torch.Tensor), f"Predicted mask must be a torch.Tensor, got {type(pred_mask)}."
        assert pred_mask.ndim in {
            2,
            3,
        }, f"Predicted mask must have shape [H, W] or [1, H, W] got shape {pred_mask.shape}."
        if pred_mask.ndim == 3:
            assert pred_mask.shape[0] == 1, f"Predicted mask must have 1 channel, got {pred_mask.shape[0]}."
            pred_mask = pred_mask.squeeze(0)
        return Mask(pred_mask, dtype=torch.bool)

    def _validate_pred_label(self, pred_label: torch.Tensor | np.ndarray | None) -> torch.Tensor | None:
        if pred_label is None:
            return None
        if not isinstance(pred_label, torch.Tensor):
            try:
                pred_label = torch.tensor(pred_label)
            except Exception as e:
                msg = "Failed to convert pred_score to a torch.Tensor."
                raise ValueError(msg) from e
        pred_label = pred_label.squeeze()
        assert pred_label.ndim == 0, f"Predicted label must be a scalar, got shape {pred_label.shape}."
        return pred_label.to(torch.bool)

    def _validate_image_path(self, image_path: str | None) -> str | None:
        if image_path is None:
            return None
        return str(image_path)


@dataclass
class ImageBatch(
    ToNumpyMixin[NumpyImageBatch],
    BatchIterateMixin[ImageItem],
    _ImageInputFields[list[str]],
    Batch[Image],
):
    """Dataclass for batches of image items in Anomalib datasets using PyTorch tensors.

    This class combines the functionality of ``ToNumpyMixin``, ``BatchIterateMixin``,
    ``_ImageInputFields``, and ``Batch`` to represent collections of image data points in Anomalib.
    It includes image-specific fields and provides methods for batch operations,
    iteration over individual items, and conversion to numpy format.

    The class is designed to work with PyTorch tensors and includes fields for
    batches of image data, ground truth labels and masks, anomaly maps, and related metadata.

    Examples:
        >>> batch = ImageBatch(
        ...     image=torch.rand(32, 3, 224, 224),
        ...     gt_label=torch.randint(0, 2, (32,)),
        ...     gt_mask=torch.rand(32, 224, 224) > 0.5,
        ...     image_path=["path/to/image_{}.jpg".format(i) for i in range(32)]
        ... )

        >>> print(batch.image.shape)
        torch.Size([32, 3, 224, 224])

        >>> for item in batch:
        ...     print(item.image.shape)
        torch.Size([3, 224, 224])

        >>> numpy_batch = batch.to_numpy()
        >>> print(type(numpy_batch))
        <class 'anomalib.dataclasses.numpy.NumpyImageBatch'>
    """

    item_class = ImageItem
    numpy_class = NumpyImageBatch

    def _validate_image(self, image: Image) -> Image:
        assert isinstance(image, torch.Tensor), f"Image must be a torch.Tensor, got {type(image)}."
        assert image.ndim in {3, 4}, f"Image must have shape [C, H, W] or [N, C, H, W], got shape {image.shape}."
        if image.ndim == 3:
            image = image.unsqueeze(0)  # add batch dimension
        assert image.shape[1] == 3, f"Image must have 3 channels, got {image.shape[0]}."
        return Image(image, dtype=torch.float32)

    def _validate_gt_label(self, gt_label: torch.Tensor | Sequence[int] | None) -> torch.Tensor:
        if gt_label is None:
            return None
        if isinstance(gt_label, Sequence):
            gt_label = torch.tensor(gt_label)
        assert isinstance(
            gt_label,
            torch.Tensor,
        ), f"Ground truth label must be a sequence of integers or a torch.Tensor, got {type(gt_label)}."
        assert gt_label.ndim == 1, f"Ground truth label must be a 1-dimensional vector, got shape {gt_label.shape}."
        assert (
            len(gt_label) == self.batch_size
        ), f"Ground truth label must have length {self.batch_size}, got length {len(gt_label)}."
        assert not torch.is_floating_point(gt_label), f"Ground truth label must be boolean or integer, got {gt_label}."
        return gt_label.bool()

    def _validate_gt_mask(self, gt_mask: Mask | None) -> Mask | None:
        if gt_mask is None:
            return None
        assert isinstance(gt_mask, torch.Tensor), f"Ground truth mask must be a torch.Tensor, got {type(gt_mask)}."
        assert gt_mask.ndim in {
            2,
            3,
            4,
        }, f"Ground truth mask must have shape [H, W] or [N, H, W] or [N, 1, H, W] got shape {gt_mask.shape}."
        if gt_mask.ndim == 2:
            assert (
                self.batch_size == 1
            ), f"Invalid shape for gt_mask. Got mask shape {gt_mask.shape} for batch size {self.batch_size}."
            gt_mask = gt_mask.unsqueeze(0)
        if gt_mask.ndim == 3:
            assert (
                gt_mask.shape[0] == self.batch_size
            ), f"Invalid shape for gt_mask. Got mask shape {gt_mask.shape} for batch size {self.batch_size}."
        if gt_mask.ndim == 4:
            assert gt_mask.shape[1] == 1, f"Ground truth mask must have 1 channel, got {gt_mask.shape[1]}."
            gt_mask = gt_mask.squeeze(1)
        return Mask(gt_mask, dtype=torch.bool)

    def _validate_mask_path(self, mask_path: Sequence[str] | Sequence[str] | None) -> list[str] | None:
        if mask_path is None:
            return None
        assert isinstance(
            mask_path,
            Sequence,
        ), f"Mask path must be a sequence of paths or strings, got {type(mask_path)}."
        assert (
            len(mask_path) == self.batch_size
        ), f"Invalid length for mask_path. Got length {len(mask_path)} for batch size {self.batch_size}."
        return [str(path) for path in mask_path]

    def _validate_anomaly_map(self, anomaly_map: torch.Tensor | np.ndarray | None) -> torch.Tensor | None:
        if anomaly_map is None:
            return None
        if not isinstance(anomaly_map, torch.Tensor):
            try:
                anomaly_map = torch.tensor(anomaly_map)
            except Exception as e:
                msg = "Failed to convert anomaly_map to a torch.Tensor."
                raise ValueError(msg) from e
        assert anomaly_map.ndim in {
            2,
            3,
            4,
        }, f"Anomaly map must have shape [H, W] or [N, H, W] or [N, 1, H, W], got shape {anomaly_map.shape}."
        if anomaly_map.ndim == 2:
            assert (
                self.batch_size == 1
            ), f"Invalid shape for anomaly_map. Got mask shape {anomaly_map.shape} for batch size {self.batch_size}."
            anomaly_map = anomaly_map.unsqueeze(0)
        if anomaly_map.ndim == 4:
            assert anomaly_map.shape[1] == 1, f"Anomaly map must have 1 channel, got {anomaly_map.shape[1]}."
            anomaly_map = anomaly_map.squeeze(1)
        return Mask(anomaly_map, dtype=torch.float32)

    def _validate_pred_score(self, pred_score: torch.Tensor | None) -> torch.Tensor | None:
        if pred_score is None and self.anomaly_map is not None:
            return torch.amax(self.anomaly_map, dim=(-2, -1))
        return pred_score

    def _validate_pred_mask(self, pred_mask: torch.Tensor) -> torch.Tensor | None:
        return pred_mask

    def _validate_pred_label(self, pred_label: torch.Tensor) -> torch.Tensor | None:
        return pred_label

    def _validate_image_path(self, image_path: list[str]) -> list[str] | None:
        return image_path


# torch video outputs
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

    def _validate_image(self, image: Image) -> Video:
        return image

    def _validate_gt_label(self, gt_label: torch.Tensor) -> torch.Tensor:
        return gt_label

    def _validate_gt_mask(self, gt_mask: Mask) -> Mask:
        return gt_mask

    def _validate_mask_path(self, mask_path: str) -> str:
        return mask_path

    def _validate_anomaly_map(self, anomaly_map: torch.Tensor) -> torch.Tensor | None:
        return anomaly_map

    def _validate_pred_score(self, pred_score: torch.Tensor | None) -> torch.Tensor | None:
        return pred_score

    def _validate_pred_mask(self, pred_mask: torch.Tensor) -> torch.Tensor | None:
        return pred_mask

    def _validate_pred_label(self, pred_label: torch.Tensor) -> torch.Tensor | None:
        return pred_label

    def _validate_original_image(self, original_image: Video) -> Video:
        return original_image

    def _validate_video_path(self, video_path: str) -> str:
        return video_path

    def _validate_target_frame(self, target_frame: torch.Tensor) -> torch.Tensor:
        return target_frame

    def _validate_frames(self, frames: torch.Tensor) -> torch.Tensor:
        return frames

    def _validate_last_frame(self, last_frame: torch.Tensor) -> torch.Tensor:
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
    """Dataclass for batches of image items in Anomalib datasets using PyTorch tensors.

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

    def _validate_image(self, image: Image) -> Video:
        return image

    def _validate_gt_label(self, gt_label: torch.Tensor) -> torch.Tensor:
        return gt_label

    def _validate_gt_mask(self, gt_mask: Mask) -> Mask:
        return gt_mask

    def _validate_mask_path(self, mask_path: list[str]) -> list[str]:
        return mask_path

    def _validate_anomaly_map(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        return anomaly_map

    def _validate_pred_score(self, pred_score: torch.Tensor) -> torch.Tensor:
        return pred_score

    def _validate_pred_mask(self, pred_mask: torch.Tensor) -> torch.Tensor:
        return pred_mask

    def _validate_pred_label(self, pred_label: torch.Tensor) -> torch.Tensor:
        return pred_label

    def _validate_original_image(self, original_image: Video) -> Video:
        return original_image

    def _validate_video_path(self, video_path: list[str]) -> list[str]:
        return video_path

    def _validate_target_frame(self, target_frame: torch.Tensor) -> torch.Tensor:
        return target_frame

    def _validate_frames(self, frames: torch.Tensor) -> torch.Tensor:
        return frames

    def _validate_last_frame(self, last_frame: torch.Tensor) -> torch.Tensor:
        return last_frame


# depth
@dataclass
class DepthItem(
    ToNumpyMixin[NumpyImageItem],
    _DepthInputFields[torch.Tensor, str],
    DatasetItem[Image],
):
    """Dataclass for individual depth items in Anomalib datasets using PyTorch tensors.

    This class represents a single depth item in Anomalib datasets using PyTorch tensors.
    It combines the functionality of ToNumpyMixin, _DepthInputFields, and DatasetItem
    to handle depth data, including depth maps, labels, and metadata.

    Examples:
        >>> item = DepthItem(
        ...     image=torch.rand(3, 224, 224),
        ...     gt_label=torch.tensor(1),
        ...     depth_map=torch.rand(224, 224),
        ...     image_path="path/to/image.jpg",
        ...     depth_path="path/to/depth.png"
        ... )

        >>> print(item.image.shape, item.depth_map.shape)
        torch.Size([3, 224, 224]) torch.Size([224, 224])
    """

    numpy_class = NumpyImageItem

    def _validate_image(self, image: Image) -> Image:
        return image

    def _validate_gt_label(self, gt_label: torch.Tensor) -> torch.Tensor:
        return gt_label

    def _validate_gt_mask(self, gt_mask: Mask) -> Mask:
        return gt_mask

    def _validate_mask_path(self, mask_path: str) -> str:
        return mask_path

    def _validate_anomaly_map(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        return anomaly_map

    def _validate_pred_score(self, pred_score: torch.Tensor) -> torch.Tensor:
        return pred_score

    def _validate_pred_mask(self, pred_mask: torch.Tensor) -> torch.Tensor:
        return pred_mask

    def _validate_pred_label(self, pred_label: torch.Tensor) -> torch.Tensor:
        return pred_label

    def _validate_image_path(self, image_path: str) -> str:
        return image_path

    def _validate_depth_map(self, depth_map: torch.Tensor) -> torch.Tensor:
        return depth_map

    def _validate_depth_path(self, depth_path: str) -> str:
        return depth_path


@dataclass
class DepthBatch(
    BatchIterateMixin[DepthItem],
    _DepthInputFields[torch.Tensor, list[str]],
    Batch[Image],
):
    """Dataclass for batches of depth items in Anomalib datasets using PyTorch tensors.

    This class represents a batch of depth items in Anomalib datasets using PyTorch tensors.
    It combines the functionality of BatchIterateMixin, _DepthInputFields, and Batch
    to handle batches of depth data, including depth maps, labels, and metadata.

    Examples:
        >>> batch = DepthBatch(
        ...     image=torch.rand(32, 3, 224, 224),
        ...     gt_label=torch.randint(0, 2, (32,)),
        ...     depth_map=torch.rand(32, 224, 224),
        ...     image_path=["path/to/image_{}.jpg".format(i) for i in range(32)],
        ...     depth_path=["path/to/depth_{}.png".format(i) for i in range(32)]
        ... )

        >>> print(batch.image.shape, batch.depth_map.shape)
        torch.Size([32, 3, 224, 224]) torch.Size([32, 224, 224])

        >>> for item in batch:
        ...     print(item.image.shape, item.depth_map.shape)
        torch.Size([3, 224, 224]) torch.Size([224, 224])
    """

    item_class = DepthItem

    def _validate_image(self, image: Image) -> Image:
        return image

    def _validate_gt_label(self, gt_label: torch.Tensor) -> torch.Tensor:
        return gt_label

    def _validate_gt_mask(self, gt_mask: Mask) -> Mask:
        return gt_mask

    def _validate_mask_path(self, mask_path: list[str]) -> list[str]:
        return mask_path

    def _validate_anomaly_map(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        return anomaly_map

    def _validate_pred_score(self, pred_score: torch.Tensor) -> torch.Tensor:
        return pred_score

    def _validate_pred_mask(self, pred_mask: torch.Tensor) -> torch.Tensor:
        return pred_mask

    def _validate_pred_label(self, pred_label: torch.Tensor) -> torch.Tensor:
        return pred_label

    def _validate_image_path(self, image_path: list[str]) -> list[str]:
        return image_path

    def _validate_depth_map(self, depth_map: torch.Tensor) -> torch.Tensor:
        return depth_map

    def _validate_depth_path(self, depth_path: list[str]) -> list[str]:
        return depth_path
