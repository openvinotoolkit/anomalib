"""Generic dataclasses that can be implemented for different data types."""

from abc import abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np
import torch
from torchvision.tv_tensors import Image, Mask, Video

ImageT = TypeVar("ImageT", Image, Video, np.ndarray)
T = TypeVar("T", torch.Tensor, np.ndarray)
MaskT = TypeVar("MaskT", Mask, np.ndarray)
PathT = TypeVar("PathT", list[Path], Path)


@dataclass
class _InputFields(Generic[T, ImageT, MaskT, PathT]):
    """Generic dataclass that defines the standard input fields."""

    image: ImageT
    gt_label: T | None = None
    gt_mask: MaskT | None = None
    mask_path: PathT | None = None


@dataclass
class _VideoInputFields(
    Generic[T, ImageT, MaskT, PathT],
    _InputFields[T, ImageT, MaskT, PathT],
):
    """Generic dataclass that defines the video input fields."""

    original_image: ImageT | None = None

    video_path: PathT | None = None
    mask_path: PathT | None = None

    target_frame: T | None = None
    frames: T | None = None
    last_frame: T | None = None


@dataclass
class _OutputFields(Generic[T, MaskT]):
    """Generic dataclass that defines the standard output fields."""

    pred_score: T | None = None
    pred_label: T | None = None
    anomaly_map: T | None = None
    pred_mask: MaskT | None = None


@dataclass
class _GenericItem(
    Generic[T, ImageT, MaskT, PathT],
    _OutputFields[T, MaskT],
    _InputFields[T, ImageT, MaskT, PathT],
):
    """Generic dataclass for a dataset item."""


ItemT = TypeVar("ItemT", bound="_GenericItem")


@dataclass
class BatchMixin(
    Generic[ItemT],
):
    """Mixin that adds batch functionalities to a dataclass."""

    @property
    @abstractmethod
    def item_class(self) -> Callable:
        """Get the item class."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[ItemT]:
        """Iterate over the batch."""
        yield from self.items

    @property
    def items(self) -> list[ItemT]:
        """Convert the batch to a list of DatasetItem objects."""
        batch_dict = asdict(self)
        return [
            self.item_class(
                **{key: value[i] if hasattr(value, "__getitem__") else None for key, value in batch_dict.items()},
            )
            for i in range(self.batch_size)
        ]

    def __len__(self) -> int:
        """Get the batch size."""
        return self.batch_size

    @property
    def batch_size(self) -> int:
        """Get the batch size."""
        for value in asdict(self).values():
            if hasattr(value, "__len__"):
                return len(value)
        msg = "Batch size not found. Make sure the batch has at least one field."
        raise ValueError(msg)
