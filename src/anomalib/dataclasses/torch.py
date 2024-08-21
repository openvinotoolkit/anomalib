"""Dataclasses for torch inputs and outputs."""

from collections.abc import Callable, Sequence
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
        assert isinstance(image, torch.Tensor), f"Image must be a torch.Tensor, got {type(image)}."
        assert image.ndim == 3, f"Image must have shape [C, H, W], got shape {image.shape}."
        assert image.shape[0] == 3, f"Image must have 3 channels, got {image.shape[0]}."
        return Image(image, dtype=torch.float32)

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

    def _validate_gt_mask(self, gt_mask: Mask | None) -> Mask | None:
        if gt_mask is None:
            return None
        assert isinstance(gt_mask, torch.Tensor), f"Ground truth mask must be a torch.Tensor, got {type(gt_mask)}."
        assert gt_mask.ndim in [
            2,
            3,
        ], f"Ground truth mask must have shape [H, W] or [1, H, W] got shape {gt_mask.shape}."
        if gt_mask.ndim == 3:
            assert gt_mask.shape[0] == 1, f"Ground truth mask must have 1 channel, got {gt_mask.shape[0]}."
            gt_mask = gt_mask.squeeze(0)
        return Mask(gt_mask, dtype=torch.bool)

    def _validate_mask_path(self, mask_path: Path | None) -> Path | None:
        if mask_path is None:
            return None
        return Path(mask_path)


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
        assert isinstance(image, torch.Tensor), f"Image must be a torch.Tensor, got {type(image)}."
        assert image.ndim in [3, 4], f"Image must have shape [C, H, W] or [N, C, H, W], got shape {image.shape}."
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
        assert gt_mask.ndim in [
            2,
            3,
            4,
        ], f"Ground truth mask must have shape [H, W] or [N, H, W] or [N, 1, H, W] got shape {gt_mask.shape}."
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

    def _validate_mask_path(self, mask_path: Sequence[Path] | Sequence[str] | None) -> list[Path] | None:
        if mask_path is None:
            return None
        assert isinstance(
            mask_path,
            Sequence,
        ), f"Mask path must be a sequence of paths or strings, got {type(mask_path)}."
        assert (
            len(mask_path) == self.batch_size
        ), f"Invalid length for mask_path. Got length {len(mask_path)} for batch size {self.batch_size}."
        return [Path(path) for path in mask_path]


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
