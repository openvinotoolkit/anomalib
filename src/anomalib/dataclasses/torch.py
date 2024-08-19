"""Dataclasses for torch inputs and outputs."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Generic, NamedTuple, TypeVar

import torch
from torchvision.tv_tensors import Image, Mask, Video

from .generic import _GenericBatch, _GenericItem, _InputFields, _OutputFields, _VideoInputFields
from .numpy import (
    NumpyImageOutputBatch,
    NumpyImageOutputItem,
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


# torch image outputs
@dataclass
class TorchImageOutputItem(
    ToNumpyMixin[NumpyImageOutputItem],
    _GenericItem,
    _OutputFields[torch.Tensor, Mask],
    _InputFields[torch.Tensor, Image, Mask, Path],
):
    """Dataclass for torch image output item."""

    @property
    def numpy_class(self) -> Callable:
        """Get the numpy equivalent of the torch dataclass."""
        return NumpyImageOutputItem


@dataclass
class TorchImageOutputBatch(
    ToNumpyMixin[NumpyImageOutputBatch],
    _GenericBatch[TorchImageOutputItem],
    _OutputFields[torch.Tensor, Mask],
    _InputFields[torch.Tensor, Image, Mask, list[Path]],
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
    _GenericItem,
    _OutputFields[torch.Tensor, Mask],
    _VideoInputFields[torch.Tensor, Video, Mask, Path],
):
    """Dataclass for torch video output item."""

    @property
    def numpy_class(self) -> Callable:
        """Get the numpy equivalent of the torch dataclass."""
        return NumpyVideoOutputItem


@dataclass
class TorchVideoOutputBatch(
    ToNumpyMixin[NumpyVideoOutputBatch],
    _GenericBatch[TorchVideoOutputItem],
    _OutputFields[torch.Tensor, Mask],
    _VideoInputFields[torch.Tensor, Video, Mask, list[Path]],
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
