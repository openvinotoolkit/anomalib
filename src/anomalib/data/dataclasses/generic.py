"""Generic dataclasses that can be implemented for different data types.

This module provides a set of generic dataclasses and mixins that can be used
to define and validate various types of data fields used in Anomalib.
The dataclasses are designed to be flexible and extensible, allowing for easy
customization and validation of input and output data.

The module contains several key components:

- Field descriptors for validation
- Base input field classes for images, videos and depth data
- Output field classes for predictions
- Mixins for updating and batch iteration
- Generic item and batch classes

Example:
    >>> from anomalib.data.dataclasses import _InputFields
    >>> from torchvision.tv_tensors import Image, Mask
    >>>
    >>> class MyInput(_InputFields[int, Image, Mask, str]):
    ...     def validate_image(self, image):
    ...         return image
    ...     # Implement other validation methods
    ...
    >>> input_data = MyInput(
    ...     image=torch.rand(3,224,224),
    ...     gt_label=1,
    ...     gt_mask=None,
    ...     mask_path=None
    ... )
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from types import NoneType
from typing import Any, ClassVar, Generic, TypeVar, get_args, get_type_hints

import numpy as np
import torch
from torch import tensor
from torch.utils.data import default_collate
from torchvision.transforms.v2.functional import resize
from torchvision.tv_tensors import Image, Mask, Video

ImageT = TypeVar("ImageT", Image, Video, np.ndarray)
T = TypeVar("T", torch.Tensor, np.ndarray)
MaskT = TypeVar("MaskT", Mask, np.ndarray)
PathT = TypeVar("PathT", list[str], str)


Instance = TypeVar("Instance")
Value = TypeVar("Value")


class FieldDescriptor(Generic[Value]):
    """Descriptor for Anomalib's dataclass fields.

    Using a descriptor ensures that the values of dataclass fields can be
    validated before being set. This allows validation of the input data not
    only when it is first set, but also when it is updated.

    Args:
        validator_name: Name of the validator method to call when setting value.
            Defaults to ``None``.
        default: Default value for the field. Defaults to ``None``.

    Example:
        >>> class MyClass:
        ...     field = FieldDescriptor(validator_name="validate_field")
        ...     def validate_field(self, value):
        ...         return value
        ...
        >>> obj = MyClass()
        >>> obj.field = 42
        >>> obj.field
        42
    """

    def __init__(self, validator_name: str | None = None, default: Value | None = None) -> None:
        """Initialize the descriptor."""
        self.validator_name = validator_name
        self.default = default

    def __set_name__(self, owner: type[Instance], name: str) -> None:
        """Set the name of the descriptor.

        Args:
            owner: Class that owns the descriptor
            name: Name of the descriptor in the owner class
        """
        self.name = name

    def __get__(self, instance: Instance | None, owner: type[Instance]) -> Value | None:
        """Get the value of the descriptor.

        Args:
            instance: Instance the descriptor is accessed from
            owner: Class that owns the descriptor

        Returns:
            Default value if instance is None, otherwise the stored value

        Raises:
            AttributeError: If no default value and field is not optional
        """
        if instance is None:
            if self.default is not None or self.is_optional(owner):
                return self.default
            msg = f"No default attribute value specified for field '{self.name}'."
            raise AttributeError(msg)
        return instance.__dict__[self.name]

    def __set__(self, instance: object, value: Value) -> None:
        """Set the value of the descriptor.

        First calls the validator method if available, then sets the value.

        Args:
            instance: Instance to set the value on
            value: Value to set
        """
        if self.validator_name is not None:
            validator = getattr(instance, self.validator_name)
            value = validator(value)
        instance.__dict__[self.name] = value

    def get_types(self, owner: type[Instance]) -> tuple[type, ...]:
        """Get the types of the descriptor.

        Args:
            owner: Class that owns the descriptor

        Returns:
            Tuple of valid types for this field

        Raises:
            TypeError: If types cannot be determined
        """
        try:
            types = get_args(get_type_hints(owner)[self.name])
            return get_args(types[0]) if hasattr(types[0], "__args__") else (types[0],)
        except (KeyError, TypeError, AttributeError) as e:
            msg = f"Unable to determine types for {self.name} in {owner}"
            raise TypeError(msg) from e

    def is_optional(self, owner: type[Instance]) -> bool:
        """Check if the descriptor is optional.

        Args:
            owner: Class that owns the descriptor

        Returns:
            True if field can be None, False otherwise
        """
        return NoneType in self.get_types(owner)


@dataclass
class _InputFields(Generic[T, ImageT, MaskT, PathT], ABC):
    """Generic dataclass that defines the standard input fields for Anomalib.

    This abstract base class provides a structure for input data used in Anomalib.
    It defines common fields used across various anomaly detection tasks and data
    types.

    Attributes:
        image: Input image or video
        gt_label: Ground truth label
        gt_mask: Ground truth segmentation mask
        mask_path: Path to mask file

    Example:
        >>> class MyInput(_InputFields[int, Image, Mask, str]):
        ...     def validate_image(self, image):
        ...         return image
        ...     # Implement other validation methods
        ...
        >>> input_data = MyInput(
        ...     image=torch.rand(3,224,224),
        ...     gt_label=1,
        ...     gt_mask=None,
        ...     mask_path=None
        ... )
    """

    image: FieldDescriptor[ImageT] = FieldDescriptor(validator_name="validate_image")
    gt_label: FieldDescriptor[T | None] = FieldDescriptor(validator_name="validate_gt_label")
    gt_mask: FieldDescriptor[MaskT | None] = FieldDescriptor(validator_name="validate_gt_mask")
    mask_path: FieldDescriptor[PathT | None] = FieldDescriptor(validator_name="validate_mask_path")

    @staticmethod
    @abstractmethod
    def validate_image(image: ImageT) -> ImageT:
        """Validate the image.

        Args:
            image: Input image to validate

        Returns:
            Validated image

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def validate_gt_mask(gt_mask: MaskT) -> MaskT | None:
        """Validate the ground truth mask.

        Args:
            gt_mask: Ground truth mask to validate

        Returns:
            Validated mask or None

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def validate_mask_path(mask_path: PathT) -> PathT | None:
        """Validate the mask path.

        Args:
            mask_path: Path to mask file to validate

        Returns:
            Validated path or None

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def validate_gt_label(gt_label: T) -> T | None:
        """Validate the ground truth label.

        Args:
            gt_label: Ground truth label to validate

        Returns:
            Validated label or None

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError


@dataclass
class _ImageInputFields(Generic[PathT], ABC):
    """Generic dataclass for image-specific input fields in Anomalib.

    This class extends standard input fields with an ``image_path`` attribute for
    image-based anomaly detection tasks.

    Attributes:
        image_path: Path to input image file

    Example:
        >>> class MyImageInput(_ImageInputFields[str]):
        ...     def validate_image_path(self, path):
        ...         return path
        ...
        >>> input_data = MyImageInput(image_path="path/to/image.jpg")
    """

    image_path: FieldDescriptor[PathT | None] = FieldDescriptor(validator_name="validate_image_path")

    @staticmethod
    @abstractmethod
    def validate_image_path(image_path: PathT) -> PathT | None:
        """Validate the image path.

        Args:
            image_path: Path to validate

        Returns:
            Validated path or None

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError


@dataclass
class _VideoInputFields(Generic[T, ImageT, MaskT, PathT], ABC):
    """Generic dataclass that defines the video input fields for Anomalib.

    This class extends standard input fields with attributes specific to
    video-based anomaly detection tasks.

    Attributes:
        original_image: Original frame from video
        video_path: Path to input video file
        target_frame: Frame number to process
        frames: Sequence of video frames
        last_frame: Last frame in sequence

    Example:
        >>> class MyVideoInput(_VideoInputFields[int, Image, Mask, str]):
        ...     def validate_original_image(self, image):
        ...         return image
        ...     # Implement other validation methods
        ...
        >>> input_data = MyVideoInput(
        ...     original_image=torch.rand(3,224,224),
        ...     video_path="video.mp4",
        ...     target_frame=10,
        ...     frames=None,
        ...     last_frame=None
        ... )
    """

    original_image: FieldDescriptor[ImageT | None] = FieldDescriptor(validator_name="validate_original_image")
    video_path: FieldDescriptor[PathT | None] = FieldDescriptor(validator_name="validate_video_path")
    target_frame: FieldDescriptor[T | None] = FieldDescriptor(validator_name="validate_target_frame")
    frames: FieldDescriptor[T | None] = FieldDescriptor(validator_name="validate_frames")
    last_frame: FieldDescriptor[T | None] = FieldDescriptor(validator_name="validate_last_frame")

    @staticmethod
    @abstractmethod
    def validate_original_image(original_image: ImageT) -> ImageT | None:
        """Validate the original image.

        Args:
            original_image: Image to validate

        Returns:
            Validated image or None

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def validate_video_path(video_path: PathT) -> PathT | None:
        """Validate the video path.

        Args:
            video_path: Path to validate

        Returns:
            Validated path or None

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def validate_target_frame(target_frame: T) -> T | None:
        """Validate the target frame.

        Args:
            target_frame: Frame number to validate

        Returns:
            Validated frame number or None

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def validate_frames(frames: T) -> T | None:
        """Validate the frames.

        Args:
            frames: Frame sequence to validate

        Returns:
            Validated frames or None

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def validate_last_frame(last_frame: T) -> T | None:
        """Validate the last frame.

        Args:
            last_frame: Frame to validate

        Returns:
            Validated frame or None

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError


@dataclass
class _DepthInputFields(Generic[T, PathT], _ImageInputFields[PathT], ABC):
    """Generic dataclass that defines the depth input fields for Anomalib.

    This class extends standard input fields with depth-specific attributes for
    depth-based anomaly detection tasks.

    Attributes:
        depth_map: Depth map image
        depth_path: Path to depth map file

    Example:
        >>> class MyDepthInput(_DepthInputFields[torch.Tensor, str]):
        ...     def validate_depth_map(self, depth):
        ...         return depth
        ...     def validate_depth_path(self, path):
        ...         return path
        ...     # Implement other validation methods
        ...
        >>> input_data = MyDepthInput(
        ...     image_path="rgb.jpg",
        ...     depth_map=torch.rand(224,224),
        ...     depth_path="depth.png"
        ... )
    """

    depth_map: FieldDescriptor[T | None] = FieldDescriptor(validator_name="validate_depth_map")
    depth_path: FieldDescriptor[PathT | None] = FieldDescriptor(validator_name="validate_depth_path")

    @staticmethod
    @abstractmethod
    def validate_depth_map(depth_map: ImageT) -> ImageT | None:
        """Validate the depth map.

        Args:
            depth_map: Depth map to validate

        Returns:
            Validated depth map or None

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def validate_depth_path(depth_path: PathT) -> PathT | None:
        """Validate the depth path.

        Args:
            depth_path: Path to validate

        Returns:
            Validated path or None

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError


@dataclass
class _OutputFields(Generic[T, MaskT, PathT], ABC):
    """Generic dataclass that defines the standard output fields for Anomalib.

    This class defines the standard output fields used in Anomalib, including
    anomaly maps, predicted scores, masks, and labels.

    Attributes:
        anomaly_map: Predicted anomaly heatmap
        pred_score: Predicted anomaly score
        pred_mask: Predicted segmentation mask
        pred_label: Predicted label
        explanation: Path to explanation visualization

    Example:
        >>> class MyOutput(_OutputFields[float, Mask, str]):
        ...     def validate_anomaly_map(self, amap):
        ...         return amap
        ...     # Implement other validation methods
        ...
        >>> output = MyOutput(
        ...     anomaly_map=torch.rand(224,224),
        ...     pred_score=0.7,
        ...     pred_mask=None,
        ...     pred_label=1,
        ...     explanation=None
        ... )
    """

    anomaly_map: FieldDescriptor[MaskT | None] = FieldDescriptor(validator_name="validate_anomaly_map")
    pred_score: FieldDescriptor[T | None] = FieldDescriptor(validator_name="validate_pred_score")
    pred_mask: FieldDescriptor[MaskT | None] = FieldDescriptor(validator_name="validate_pred_mask")
    pred_label: FieldDescriptor[T | None] = FieldDescriptor(validator_name="validate_pred_label")
    explanation: FieldDescriptor[PathT | None] = FieldDescriptor(validator_name="validate_explanation")

    @staticmethod
    @abstractmethod
    def validate_anomaly_map(anomaly_map: MaskT) -> MaskT | None:
        """Validate the anomaly map.

        Args:
            anomaly_map: Anomaly map to validate

        Returns:
            Validated anomaly map or None

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def validate_pred_score(pred_score: T) -> T | None:
        """Validate the predicted score.

        Args:
            pred_score: Score to validate

        Returns:
            Validated score or None

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def validate_pred_mask(pred_mask: MaskT) -> MaskT | None:
        """Validate the predicted mask.

        Args:
            pred_mask: Mask to validate

        Returns:
            Validated mask or None

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def validate_pred_label(pred_label: T) -> T | None:
        """Validate the predicted label.

        Args:
            pred_label: Label to validate

        Returns:
            Validated label or None

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def validate_explanation(explanation: PathT) -> PathT | None:
        """Validate the explanation.

        Args:
            explanation: Explanation to validate

        Returns:
            Validated explanation or None

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError


@dataclass
class UpdateMixin:
    """Mixin class for dataclasses that allows for in-place replacement of attrs.

    This mixin provides methods for updating dataclass instances in place or by
    creating a new instance.

    Example:
        >>> @dataclass
        ... class MyItem(UpdateMixin):
        ...     field1: int
        ...     field2: str
        ...
        >>> item = MyItem(field1=1, field2="a")
        >>> item.update(field1=2)  # In-place update
        >>> item.field1
        2
        >>> new_item = item.update(in_place=False, field2="b")
        >>> new_item.field2
        'b'
    """

    def update(self, in_place: bool = True, **changes) -> Any:  # noqa: ANN401
        """Replace fields in place and call __post_init__ to reinitialize.

        Args:
            in_place: Whether to modify in place or return new instance
            **changes: Field names and new values to update

        Returns:
            Updated instance (self if in_place=True, new instance otherwise)

        Raises:
            TypeError: If instance is not a dataclass
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
    _OutputFields[T, MaskT, PathT],
    _InputFields[T, ImageT, MaskT, PathT],
):
    """Generic dataclass for a single item in Anomalib datasets.

    This class combines input and output fields for anomaly detection tasks.
    It inherits from ``_InputFields`` for standard input data and
    ``_OutputFields`` for prediction results.

    Example:
        >>> class MyItem(_GenericItem[int, Image, Mask, str]):
        ...     def validate_image(self, image):
        ...         return image
        ...     # Implement other validation methods
        ...
        >>> item = MyItem(
        ...     image=torch.rand(3,224,224),
        ...     gt_label=0,
        ...     pred_score=0.3,
        ...     anomaly_map=torch.rand(224,224)
        ... )
        >>> item.update(pred_score=0.8)
    """


@dataclass
class _GenericBatch(
    UpdateMixin,
    Generic[T, ImageT, MaskT, PathT],
    _OutputFields[T, MaskT, PathT],
    _InputFields[T, ImageT, MaskT, PathT],
):
    """Generic dataclass for a batch of items in Anomalib datasets.

    This class represents a batch of data items, combining both input and output
    fields for anomaly detection tasks.

    Example:
        >>> class MyBatch(_GenericBatch[int, Image, Mask, str]):
        ...     def validate_image(self, image):
        ...         return image
        ...     # Implement other validation methods
        ...
        >>> batch = MyBatch(
        ...     image=torch.rand(32,3,224,224),
        ...     gt_label=torch.zeros(32),
        ...     pred_score=torch.rand(32)
        ... )
    """


ItemT = TypeVar("ItemT", bound="_GenericItem")


@dataclass
class BatchIterateMixin(Generic[ItemT]):
    """Mixin class for iterating over batches of items in Anomalib datasets.

    This class provides functionality to iterate over individual items within a
    batch and convert batches to lists of items.

    Attributes:
        item_class: Class to use for individual items in the batch

    Example:
        >>> @dataclass
        ... class MyBatch(BatchIterateMixin):
        ...     item_class = MyItem
        ...     data: torch.Tensor
        ...
        >>> batch = MyBatch(data=torch.rand(32,3,224,224))
        >>> for item in batch:
        ...     process_item(item)
        >>> items = batch.items  # Convert to list of items
    """

    item_class: ClassVar[Callable]

    def __init_subclass__(cls, **kwargs) -> None:
        """Ensure that the subclass has the required attributes.

        Args:
            **kwargs: Additional keyword arguments

        Raises:
            AttributeError: If item_class is not defined
        """
        super().__init_subclass__(**kwargs)
        if not (hasattr(cls, "item_class") or issubclass(cls, ABC)):
            msg = f"{cls.__name__} must have an 'item_class' attribute."
            raise AttributeError(msg)

    def __iter__(self) -> Iterator[ItemT]:
        """Iterate over the batch.

        Yields:
            Individual items from the batch
        """
        yield from self.items

    @property
    def items(self) -> list[ItemT]:
        """Convert the batch to a list of DatasetItem objects.

        Returns:
            List of individual items from the batch
        """
        batch_dict = asdict(self)
        return [
            self.item_class(
                **{key: value[i] if hasattr(value, "__getitem__") else None for key, value in batch_dict.items()},
            )
            for i in range(self.batch_size)
        ]

    def __len__(self) -> int:
        """Get the batch size.

        Returns:
            Number of items in batch
        """
        return self.batch_size

    @property
    def batch_size(self) -> int:
        """Get the batch size.

        Returns:
            Number of items in batch

        Raises:
            AttributeError: If image attribute is not set
        """
        try:
            image = getattr(self, "image")  # noqa: B009
            return len(image)
        except (KeyError, AttributeError) as e:
            msg = "Cannot determine batch size because 'image' attribute not set."
            raise AttributeError(msg) from e

    @classmethod
    def collate(cls: type["BatchIterateMixin"], items: list[ItemT]) -> "BatchIterateMixin":
        """Convert a list of DatasetItem objects to a Batch object.

        Args:
            items: List of items to collate into a batch

        Returns:
            New batch containing the items
        """
        keys = [key for key, value in asdict(items[0]).items() if value is not None]

        # Check if all images have the same shape. If not, resize before collating
        im_shapes = torch.vstack([tensor(item.image.shape) for item in items if item.image is not None])[..., 1:]
        if torch.unique(im_shapes, dim=0).size(0) != 1:  # check if batch has heterogeneous shapes
            target_shape = im_shapes[
                torch.unravel_index(im_shapes.argmax(), im_shapes.shape)[0],
                :,
            ]  # shape of image with largest H or W
            for item in items:
                for key in keys:
                    value = getattr(item, key)
                    if isinstance(value, Image | Mask):
                        setattr(item, key, resize(value, target_shape))

        # collate the batch
        out_dict = {key: default_collate([getattr(item, key) for item in items]) for key in keys}
        return cls(**out_dict)
