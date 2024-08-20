"""Generic dataclasses that can be implemented for different data types."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from types import NoneType
from typing import ClassVar, Generic, TypeVar, get_args, get_type_hints

import numpy as np
import torch
from torch.utils.data import default_collate
from torchvision.tv_tensors import Image, Mask, Video

ImageT = TypeVar("ImageT", Image, Video, np.ndarray)
T = TypeVar("T", torch.Tensor, np.ndarray)
MaskT = TypeVar("MaskT", Mask, np.ndarray)
PathT = TypeVar("PathT", list[Path], Path)


Instance = TypeVar("Instance")
Value = TypeVar("Value")


class FieldDescriptor(
    Generic[Value],
):
    """Descriptor for Anomalib's dataclass fields.

    Using a descriptor ensures that the values of dataclass fields can be validated before being set.
    This allows validation of the input data not only when it is first set, but also when it is updated.
    """

    def __init__(self, validator_name: str | None = None, default: Value | None = None) -> None:
        """Initialize the descriptor."""
        self.validator_name = validator_name
        self.default = default

    def __set_name__(self, owner: type[Instance], name: str) -> None:
        """Set the name of the descriptor."""
        self.name = name

    def __get__(self, instance: Instance | None, owner: type[Instance]) -> Value | None:
        """Get the value of the descriptor.

        Returns:
            - The default value if available and if the instance is None (method is called from class).
            - The value of the attribute if the instance is not None (method is called from instance).
        """
        if instance is None:
            if self.default is not None or self.is_optional(owner):
                return self.default
            msg = f"No default attribute value specified for field '{self.name}'."
            raise AttributeError(msg)
        return instance.__dict__[self.name]

    def __set__(self, instance: Instance, value: Value) -> None:
        """Set the value of the descriptor.

        First calls the validator method if available, then sets the value of the attribute.
        """
        if self.validator_name is not None:
            validator = getattr(instance, self.validator_name)
            value = validator(value)
        instance.__dict__[self.name] = value

    def get_types(self, owner: type[Instance]) -> tuple[type, ...]:
        """Get the types of the descriptor."""
        try:
            types = get_args(get_type_hints(owner)[self.name])
            return get_args(types[0]) if hasattr(types[0], "__args__") else (types[0],)
        except (KeyError, TypeError, AttributeError) as e:
            msg = f"Unable to determine types for {self.name} in {owner}"
            raise TypeError(msg) from e

    def is_optional(self, owner: type[Instance]) -> bool:
        """Check if the descriptor is optional."""
        return NoneType in self.get_types(owner)


@dataclass
class _InputFields(Generic[T, ImageT, MaskT, PathT], ABC):
    """Generic dataclass that defines the standard input fields."""

    image: FieldDescriptor[ImageT] = FieldDescriptor(validator_name="_validate_image")
    gt_label: FieldDescriptor[T | None] = FieldDescriptor(validator_name="_validate_gt_label")
    gt_mask: FieldDescriptor[MaskT | None] = FieldDescriptor(validator_name="_validate_gt_mask")
    mask_path: FieldDescriptor[PathT | None] = FieldDescriptor(validator_name="_validate_mask_path")

    @abstractmethod
    def _validate_image(self, image: ImageT) -> ImageT:
        """Validate the image."""
        raise NotImplementedError

    @abstractmethod
    def _validate_gt_mask(self, gt_mask: MaskT) -> MaskT:
        """Validate the ground truth mask."""
        raise NotImplementedError

    @abstractmethod
    def _validate_mask_path(self, mask_path: PathT) -> PathT:
        """Validate the mask path."""
        raise NotImplementedError

    @abstractmethod
    def _validate_gt_label(self, gt_label: T) -> T:
        """Validate the ground truth label."""
        raise NotImplementedError


@dataclass
class _ImageInputFields(
    Generic[T, ImageT, MaskT, PathT],
    _InputFields[T, ImageT, MaskT, PathT],
):
    """Generic dataclass that defines the image input fields."""


@dataclass
class _VideoInputFields(
    Generic[T, ImageT, MaskT, PathT],
    _InputFields[T, ImageT, MaskT, PathT],
):
    """Generic dataclass that defines the video input fields."""

    original_image: ImageT | None = None

    video_path: PathT | None = None

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
class _GenericItem:
    """Generic dataclass for a dataset item."""


ItemT = TypeVar("ItemT", bound="_GenericItem")


@dataclass
class _GenericBatch(Generic[ItemT]):
    """Generic dataclass for a batch."""

    item_class: ClassVar[Callable]

    def __init_subclass__(cls, **kwargs) -> None:
        """Ensure that the subclass has the required attributes."""
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "item_class"):
            msg = f"{cls.__name__} must have an 'item_class' attribute."
            raise AttributeError(msg)

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

    @classmethod
    def collate(cls: type["_GenericBatch"], items: list[ItemT]) -> "_GenericBatch":
        """Convert a list of DatasetItem objects to a Batch object."""
        keys = [key for key, value in asdict(items[0]).items() if value is not None]
        out_dict = {key: default_collate([getattr(item, key) for item in items]) for key in keys}
        return cls(**out_dict)
