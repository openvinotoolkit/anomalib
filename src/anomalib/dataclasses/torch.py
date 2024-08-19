"""Dataclasses for torch inputs and outputs."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Generic, NamedTuple, TypeVar

import torch
from torchvision.tv_tensors import Image, Mask, Video

from .generic import BatchMixin, ImageT, PathT, _GenericItem, _InputFields, _OutputFields, _VideoInputFields
from .numpy import (
    NumpyImageInputBatch,
    NumpyImageInputItem,
    NumpyImageOutputBatch,
    NumpyImageOutputItem,
    NumpyVideoInputBatch,
    NumpyVideoInputItem,
    NumpyVideoOutputBatch,
    NumpyVideoOutputItem,
)

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
    ABC,
):
    """Mixin for converting torch-based dataclasses to numpy."""

    @property
    @abstractmethod
    def numpy_class(self) -> Callable:
        """Get the numpy class.

        This property should be implemented in the subclass.
        """
        raise NotImplementedError

    def to_numpy(self) -> NumpyT:
        """Convert the batch to a NumpyBatch object."""
        batch_dict = asdict(self)
        for key, value in batch_dict.items():
            if isinstance(value, torch.Tensor):
                batch_dict[key] = value.cpu().numpy()
        return self.numpy_class(
            **batch_dict,
        )


# torch inputs
@dataclass
class _TorchInput(
    Generic[ImageT, PathT],
    _InputFields[torch.Tensor, ImageT, Mask, PathT],
):
    """Dataclass for torch image inputs.

    Common class for both single item and batch, and any modality (image, video, etc.).
    """


@dataclass
class _TorchInputItem(
    Generic[ImageT],
    _GenericItem[torch.Tensor, ImageT, Mask, Path],
    _TorchInput[ImageT, Path],
):
    """Dataclass for torch input item.

    Common class for any modality (image, video, etc.).
    """


@dataclass
class _TorchInputBatch(
    Generic[ImageT],
    _TorchInput[ImageT, list[Path]],
):
    """Dataclass for torch input batch.

    Common class for any modality (image, video, etc.).
    """


# torch image inputs
@dataclass
class TorchImageInputItem(
    ToNumpyMixin[NumpyImageInputItem],
    _TorchInputItem[Image],
):
    """Dataclass for torch image input item."""

    @property
    def numpy_class(self) -> Callable:
        """Get the numpy equivalent of the torch dataclass."""
        return NumpyImageInputItem


@dataclass
class TorchImageInputBatch(
    BatchMixin[TorchImageInputItem],
    ToNumpyMixin[NumpyImageInputBatch],
    _TorchInputBatch[Image],
):
    """Dataclass for torch image input batch."""

    @property
    def item_class(self) -> Callable:
        """Get the single-item equivalent of the batch class."""
        return TorchImageInputItem

    @property
    def numpy_class(self) -> Callable:
        """Get the numpy equivalent of the torch dataclass."""
        return NumpyImageInputBatch


# torch video inputs
@dataclass
class _TorchVideoInput(
    Generic[ImageT, PathT],
    _VideoInputFields[torch.Tensor, ImageT, Mask, PathT],
):
    """Dataclass for torch video input.

    Common class for both single item and batch.
    """


@dataclass
class TorchVideoInputItem(
    ToNumpyMixin[NumpyVideoInputItem],
    _TorchInputItem[Video],
    _TorchVideoInput[Video, Path],
):
    """Dataclass for torch video input item."""

    @property
    def numpy_class(self) -> Callable:
        """Get the numpy equivalent of the torch dataclass."""
        return NumpyVideoInputItem


@dataclass
class TorchVideoInputBatch(
    BatchMixin[TorchVideoInputItem],
    ToNumpyMixin[NumpyVideoInputBatch],
    _TorchInputBatch[Video],
    _TorchVideoInput[Video, list[Path]],
):
    """Dataclass for torch video input batch."""

    @property
    def item_class(self) -> Callable:
        """Get the single-item equivalent of the batch class."""
        return TorchVideoInputItem

    @property
    def numpy_class(self) -> Callable:
        """Get the numpy equivalent of the torch dataclass."""
        return NumpyVideoInputBatch


###### OUTPUTS ######


# torch outputs
@dataclass
class _TorchOutput(
    Generic[ImageT, PathT],
    _OutputFields[ImageT, Mask],
    _TorchInput[ImageT, PathT],
):
    """Dataclass for torch output.

    Common class for both single item and batch, and any modality (image, video, etc.).
    """


@dataclass
class _TorchOutputItem(
    Generic[ImageT],
    _TorchOutput[ImageT, Path],
    _TorchInputItem[ImageT],
):
    """Dataclass for torch output item.

    Common class for any modality (image, video, etc.).
    """


@dataclass
class _TorchOutputBatch(
    Generic[ImageT],
    _TorchOutput[ImageT, list[Path]],
    _TorchInputBatch[ImageT],
):
    """Dataclass for torch output batch.

    Common class for any modality (image, video, etc.).
    """


# torch image outputs
@dataclass
class TorchImageOutputItem(
    ToNumpyMixin[NumpyImageOutputItem],
    _TorchOutputItem[Image],
):
    """Dataclass for torch image output item."""

    @property
    def numpy_class(self) -> Callable:
        """Get the numpy equivalent of the torch dataclass."""
        return NumpyImageOutputItem


@dataclass
class TorchImageOutputBatch(
    ToNumpyMixin[NumpyImageOutputBatch],
    BatchMixin[TorchImageInputItem],
    _TorchOutputBatch[Image],
):
    """Dataclass for torch image output batch."""

    @property
    def item_class(self) -> Callable:
        """Get the single-item equivalent of the batch class."""
        return TorchImageOutputItem

    @property
    def numpy_class(self) -> Callable:
        """Get the numpy equivalent of the torch dataclass."""
        return NumpyImageOutputBatch


# torch video outputs
@dataclass
class TorchVideoOutputItem(
    ToNumpyMixin[NumpyVideoOutputItem],
    _TorchOutputItem[Video],
    _TorchVideoInput[Video, Path],
):
    """Dataclass for torch video output item."""

    @property
    def numpy_class(self) -> Callable:
        """Get the numpy equivalent of the torch dataclass."""
        return NumpyVideoOutputItem


@dataclass
class TorchVideoOutputBatch(
    BatchMixin[TorchVideoOutputItem],
    ToNumpyMixin[NumpyVideoOutputBatch],
    _TorchOutputBatch[Video],
    _TorchVideoInput[Video, list[Path]],
):
    """Dataclass for torch video output batch."""

    @property
    def item_class(self) -> Callable:
        """Get the single-item equivalent of the batch class."""
        return TorchVideoOutputItem

    @property
    def numpy_class(self) -> Callable:
        """Get the numpy equivalent of the torch dataclass."""
        return NumpyVideoOutputBatch
