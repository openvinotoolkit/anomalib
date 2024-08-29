"""Generic dataclasses that can be implemented for different data types."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from types import NoneType
from typing import Any, ClassVar, Generic, TypeVar, get_args, get_type_hints

import numpy as np
import torch
from torch.utils.data import default_collate
from torchvision.tv_tensors import Image, Mask, Video

ImageT = TypeVar("ImageT", Image, Video, np.ndarray)
T = TypeVar("T", torch.Tensor, np.ndarray)
MaskT = TypeVar("MaskT", Mask, np.ndarray)
PathT = TypeVar("PathT", list[str], str)


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
    def _validate_gt_mask(self, gt_mask: MaskT) -> MaskT | None:
        """Validate the ground truth mask."""
        raise NotImplementedError

    @abstractmethod
    def _validate_mask_path(self, mask_path: PathT) -> PathT | None:
        """Validate the mask path."""
        raise NotImplementedError

    @abstractmethod
    def _validate_gt_label(self, gt_label: T) -> T | None:
        """Validate the ground truth label."""
        raise NotImplementedError


@dataclass
class _ImageInputFields(
    Generic[PathT],
    ABC,
):
    """Generic dataclass that defines the image input fields."""

    image_path: FieldDescriptor[PathT | None] = FieldDescriptor(validator_name="_validate_image_path")

    @abstractmethod
    def _validate_image_path(self, image_path: PathT) -> PathT | None:
        """Validate the image path."""
        raise NotImplementedError


@dataclass
class _VideoInputFields(
    Generic[T, ImageT, MaskT, PathT],
    ABC,
):
    """Generic dataclass that defines the video input fields."""

    original_image: FieldDescriptor[ImageT | None] = FieldDescriptor(validator_name="_validate_original_image")
    video_path: FieldDescriptor[PathT | None] = FieldDescriptor(validator_name="_validate_video_path")
    target_frame: FieldDescriptor[T | None] = FieldDescriptor(validator_name="_validate_target_frame")
    frames: FieldDescriptor[T | None] = FieldDescriptor(validator_name="_validate_frames")
    last_frame: FieldDescriptor[T | None] = FieldDescriptor(validator_name="_validate_last_frame")

    @abstractmethod
    def _validate_original_image(self, original_image: ImageT) -> ImageT | None:
        """Validate the original image."""
        raise NotImplementedError

    @abstractmethod
    def _validate_video_path(self, video_path: PathT) -> PathT | None:
        """Validate the video path."""
        raise NotImplementedError

    @abstractmethod
    def _validate_target_frame(self, target_frame: T) -> T | None:
        """Validate the target frame."""
        raise NotImplementedError

    @abstractmethod
    def _validate_frames(self, frames: T) -> T | None:
        """Validate the frames."""
        raise NotImplementedError

    @abstractmethod
    def _validate_last_frame(self, last_frame: T) -> T | None:
        """Validate the last frame."""
        raise NotImplementedError


@dataclass
class _DepthInputFields(
    Generic[T, PathT],
    _ImageInputFields[PathT],
    ABC,
):
    """Generic dataclass that defines the depth input fields."""

    depth_map: FieldDescriptor[T | None] = FieldDescriptor(validator_name="_validate_depth_map")
    depth_path: FieldDescriptor[PathT | None] = FieldDescriptor(validator_name="_validate_depth_path")

    @abstractmethod
    def _validate_depth_map(self, depth_map: ImageT) -> ImageT | None:
        """Validate the depth map."""
        raise NotImplementedError

    @abstractmethod
    def _validate_depth_path(self, depth_path: PathT) -> PathT | None:
        """Validate the depth path."""
        raise NotImplementedError


@dataclass
class _OutputFields(Generic[T, MaskT], ABC):
    """Generic dataclass that defines the standard output fields."""

    anomaly_map: FieldDescriptor[MaskT | None] = FieldDescriptor(validator_name="_validate_anomaly_map")
    pred_score: FieldDescriptor[T | None] = FieldDescriptor(validator_name="_validate_pred_score")
    pred_mask: FieldDescriptor[MaskT | None] = FieldDescriptor(validator_name="_validate_pred_mask")
    pred_label: FieldDescriptor[T | None] = FieldDescriptor(validator_name="_validate_pred_label")

    @abstractmethod
    def _validate_anomaly_map(self, anomaly_map: MaskT) -> MaskT | None:
        """Validate the anomaly map."""
        raise NotImplementedError

    @abstractmethod
    def _validate_pred_score(self, pred_score: T) -> T | None:
        """Validate the predicted score."""
        raise NotImplementedError

    @abstractmethod
    def _validate_pred_mask(self, pred_mask: MaskT) -> MaskT | None:
        """Validate the predicted mask."""
        raise NotImplementedError

    @abstractmethod
    def _validate_pred_label(self, pred_label: T) -> T | None:
        """Validate the predicted label."""
        raise NotImplementedError


@dataclass
class UpdateMixin:
    """Mixin class for dataclasses that allows for in-place replacement of attributes."""

    def update(self, in_place: bool = True, **changes) -> Any:  # noqa: ANN401
        """Replace fields in place and call __post_init__ to reinitialize the instance.

        Parameters:
        changes (dict): A dictionary of field names and their new values.
        """
        if not is_dataclass(self):
            msg = "replace can only be used with dataclass instances"
            raise TypeError(msg)

        if in_place:
            for field in fields(self):
                if field.init and field.name in changes:
                    setattr(self, field.name, changes[field.name])
            if hasattr(self, "__post_init__"):
                self.__post_init__()
            return self
        return replace(self, **changes)


@dataclass
class _GenericItem(
    UpdateMixin,
    Generic[T, ImageT, MaskT, PathT],
    _OutputFields[T, MaskT],
    _InputFields[T, ImageT, MaskT, PathT],
):
    """Generic dataclass for a dataset item."""


@dataclass
class _GenericBatch(
    UpdateMixin,
    Generic[T, ImageT, MaskT, PathT],
    _OutputFields[T, MaskT],
    _InputFields[T, ImageT, MaskT, PathT],
):
    """Generic dataclass for a batch."""


ItemT = TypeVar("ItemT", bound="_GenericItem")


@dataclass
class BatchIterateMixin(Generic[ItemT]):
    """Generic dataclass for a batch."""

    item_class: ClassVar[Callable]

    def __init_subclass__(cls, **kwargs) -> None:
        """Ensure that the subclass has the required attributes."""
        super().__init_subclass__(**kwargs)
        if not (hasattr(cls, "item_class") or issubclass(cls, ABC)):
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
        try:
            image = getattr(self, "image")  # noqa: B009
            return len(image)
        except (KeyError, AttributeError) as e:
            msg = "Cannot determine batch size because 'image' attribute has not been set."
            raise AttributeError(msg) from e

    @classmethod
    def collate(cls: type["BatchIterateMixin"], items: list[ItemT]) -> "BatchIterateMixin":
        """Convert a list of DatasetItem objects to a Batch object."""
        keys = [key for key, value in asdict(items[0]).items() if value is not None]
        out_dict = {key: default_collate([getattr(item, key) for item in items]) for key in keys}
        return cls(**out_dict)
