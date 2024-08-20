"""Dataclasses for torch inputs and outputs."""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar, Generic, NamedTuple, TypeVar

import torch
from torchvision.tv_tensors import Image, Mask, Video

from .generic import _GenericBatch, _GenericItem, _InputFields, _OutputFields, _VideoInputFields
from .numpy import NumpyImageBatch, NumpyImageItem, NumpyVideoBatch, NumpyVideoItem

NumpyT = TypeVar("NumpyT")


class InferenceBatch(NamedTuple):
    """Batch for use in torch and inference models."""

    pred_score: torch.Tensor | None = None
    pred_label: torch.Tensor | None = None
    anomaly_map: torch.Tensor | None = None
    pred_mask: torch.Tensor | None = None


@dataclass
class ToNumpyMixin(
    Generic[NumpyT],
):
    """Mixin for converting torch-based dataclasses to numpy."""

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


# torch image outputs
@dataclass
class ImageItem(
    ToNumpyMixin[NumpyImageItem],
    _GenericItem,
    _OutputFields[torch.Tensor, Mask],
    _InputFields[torch.Tensor, Image, Mask, Path],
):
    """Dataclass for torch image output item."""

    numpy_class = NumpyImageItem

    def _validate_image(self, image: Image) -> Image:
        return image

    def _validate_gt_label(self, gt_label: torch.Tensor) -> torch.Tensor:
        return gt_label

    def _validate_gt_mask(self, gt_mask: Mask) -> Mask:
        return gt_mask

    def _validate_mask_path(self, mask_path: Path) -> Path:
        return mask_path


@dataclass
class ImageBatch(
    ToNumpyMixin[NumpyImageBatch],
    _GenericBatch[ImageItem],
    _OutputFields[torch.Tensor, Mask],
    _InputFields[torch.Tensor, Image, Mask, list[Path]],
):
    """Dataclass for torch image output batch."""

    item_class = ImageItem
    numpy_class = NumpyImageBatch

    def _validate_image(self, image: Image) -> Image:
        return image

    def _validate_gt_label(self, gt_label: torch.Tensor) -> torch.Tensor:
        return gt_label

    def _validate_gt_mask(self, gt_mask: Mask) -> Mask:
        return gt_mask

    def _validate_mask_path(self, mask_path: list[Path]) -> list[Path]:
        return mask_path


# torch video outputs
@dataclass
class VideoItem(
    ToNumpyMixin[NumpyVideoItem],
    _GenericItem,
    _OutputFields[torch.Tensor, Mask],
    _VideoInputFields[torch.Tensor, Video, Mask, Path],
):
    """Dataclass for torch video output item."""

    numpy_class = NumpyVideoItem

    def _validate_image(self, image: Image) -> Video:
        return image

    def _validate_gt_label(self, gt_label: torch.Tensor) -> torch.Tensor:
        return gt_label

    def _validate_gt_mask(self, gt_mask: Mask) -> Mask:
        return gt_mask

    def _validate_mask_path(self, mask_path: Path) -> Path:
        return mask_path


@dataclass
class VideoBatch(
    ToNumpyMixin[NumpyVideoBatch],
    _GenericBatch[VideoItem],
    _OutputFields[torch.Tensor, Mask],
    _VideoInputFields[torch.Tensor, Video, Mask, list[Path]],
):
    """Dataclass for torch video output batch."""

    item_class = VideoItem
    numpy_class = NumpyVideoBatch

    def _validate_image(self, image: Image) -> Video:
        return image

    def _validate_gt_label(self, gt_label: torch.Tensor) -> torch.Tensor:
        return gt_label

    def _validate_gt_mask(self, gt_mask: Mask) -> Mask:
        return gt_mask

    def _validate_mask_path(self, mask_path: list[Path]) -> list[Path]:
        return mask_path
