"""Generic dataclasses that can be implemented for different data types.

This module provides a set of generic dataclasses and mixins that can be used
to define and validate various types of data fields used in Anomalib.
The dataclasses are designed to be flexible and extensible, allowing for easy
customization and validation of input and output data.
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
from torch.utils.data import default_collate
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

    Attributes:
        validator_name (str | None): The name of the validator method to be
            called when setting the value.
            Defaults to ``None``.
        default (Value | None): The default value for the field.
            Defaults to ``None``.
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

    def __set__(self, instance: object, value: Value) -> None:
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
    """Generic dataclass that defines the standard input fields for Anomalib.

    This abstract base class provides a structure for input data used in Anomalib,
    a library for anomaly detection in images and videos. It defines common fields
    used across various anomaly detection tasks and data types in Anomalib.

    Subclasses must implement the abstract validation methods to define the
    specific validation logic for each field based on the requirements of different
    Anomalib models and data processing pipelines.

    Examples:
        Assuming a concrete implementation `DummyInput`:

        >>> class DummyInput(_InputFields[int, Image, Mask, str]):
        ...     # Implement actual validation

        >>> # Create an input instance
        >>> input_item = DummyInput(
        ...     image=torch.rand(3, 224, 224),
        ...     gt_label=1,
        ...     gt_mask=torch.rand(224, 224) > 0.5,
        ...     mask_path="path/to/mask.png"
        ... )

        >>> # Access fields
        >>> image = input_item.image
        >>> label = input_item.gt_label

    Note:
        This is an abstract base class and is not intended to be instantiated
        directly. Concrete subclasses should implement all required validation
        methods.
    """

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
class _ImageInputFields(Generic[PathT], ABC):
    """Generic dataclass for image-specific input fields in Anomalib.

    This class extends standard input fields with an ``image_path`` attribute for
    image-based anomaly detection tasks. It allows Anomalib to work efficiently
    with disk-stored image datasets, facilitating custom data loading strategies.

    The ``image_path`` field uses a ``FieldDescriptor`` with a validation method.
    Subclasses must implement ``_validate_image_path`` to ensure path validity
    according to specific Anomalib model or dataset requirements.

    This class is designed to complement ``_InputFields`` for comprehensive
    image-based anomaly detection input in Anomalib.

    Examples:
        Assuming a concrete implementation ``DummyImageInput``:
        >>> class DummyImageInput(_ImageInputFields):
        ...     def _validate_image_path(self, image_path):
        ...         return image_path  # Implement actual validation
        ...     # Implement other required methods

        >>> # Create an image input instance
        >>> image_input = DummyImageInput(
        ...     image_path="path/to/image.jpg"
        ... )

        >>> # Access image-specific field
        >>> path = image_input.image_path

    Note:
        This is an abstract base class and is not intended to be instantiated
        directly. Concrete subclasses should implement all required validation
        methods.
    """

    image_path: FieldDescriptor[PathT | None] = FieldDescriptor(validator_name="_validate_image_path")

    @abstractmethod
    def _validate_image_path(self, image_path: PathT) -> PathT | None:
        """Validate the image path."""
        raise NotImplementedError


@dataclass
class _VideoInputFields(Generic[T, ImageT, MaskT, PathT], ABC):
    """Generic dataclass that defines the video input fields for Anomalib.

    This class extends standard input fields with attributes specific to video-based
    anomaly detection tasks. It includes fields for original images, video paths,
    target frames, frame sequences, and last frames.

    Each field uses a ``FieldDescriptor`` with a corresponding validation method.
    Subclasses must implement these abstract validation methods to ensure data
    consistency with Anomalib's video processing requirements.

    This class is designed to work alongside other input field classes to provide
    comprehensive support for video-based anomaly detection in Anomalib.

    Examples:
        Assuming a concrete implementation ``DummyVideoInput``:

        >>> class DummyVideoInput(_VideoInputFields):
        ...     def _validate_original_image(self, original_image):
        ...         return original_image  # Implement actual validation
        ...     # Implement other required methods

        >>> # Create a video input instance
        >>> video_input = DummyVideoInput(
        ...     original_image=torch.rand(3, 224, 224),
        ...     video_path="path/to/video.mp4",
        ...     target_frame=10,
        ...     frames=torch.rand(3, 224, 224),
        ...     last_frame=torch.rand(3, 224, 224)
        ... )

        >>> # Access video-specific fields
        >>> original_image = video_input.original_image
        >>> path = video_input.video_path
        >>> target_frame = video_input.target_frame
        >>> frames = video_input.frames
        >>> last_frame = video_input.last_frame

    Note:
        This is an abstract base class and is not intended to be instantiated
        directly. Concrete subclasses should implement all required validation
        methods.
    """

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
class _DepthInputFields(Generic[T, PathT], _ImageInputFields[PathT], ABC):
    """Generic dataclass that defines the depth input fields for Anomalib.

    This class extends the standard input fields with a ``depth_map`` and
    ``depth_path`` attribute for depth-based anomaly detection tasks. It allows
    Anomalib to work efficiently with depth-based anomaly detection tasks,
    facilitating custom data loading strategies.

    The ``depth_map`` and ``depth_path`` fields use a ``FieldDescriptor`` with
    corresponding validation methods. Subclasses must implement these abstract
    validation methods to ensure data consistency with Anomalib's depth processing
    requirements.

    Examples:
        Assuming a concrete implementation ``DummyDepthInput``:

        >>> class DummyDepthInput(_DepthInputFields):
        ...     def _validate_depth_map(self, depth_map):
        ...         return depth_map  # Implement actual validation
        ...     def _validate_depth_path(self, depth_path):
        ...         return depth_path  # Implement actual validation
        ...     # Implement other required methods

        >>> # Create a depth input instance
        >>> depth_input = DummyDepthInput(
        ...     image_path="path/to/image.jpg",
        ...     depth_map=torch.rand(224, 224),
        ...     depth_path="path/to/depth.png"
        ... )

        >>> # Access depth-specific fields
        >>> depth_map = depth_input.depth_map
        >>> depth_path = depth_input.depth_path

    Note:
        This is an abstract base class and is not intended to be instantiated
        directly. Concrete subclasses should implement all required validation
        methods.
    """

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
    """Generic dataclass that defines the standard output fields for Anomalib.

    This class defines the standard output fields used in Anomalib, including
    anomaly maps, predicted scores, predicted masks, and predicted labels.

    Each field uses a ``FieldDescriptor`` with a corresponding validation method.
    Subclasses must implement these abstract validation methods to ensure data
    consistency with Anomalib's anomaly detection tasks.

    Examples:
        Assuming a concrete implementation ``DummyOutput``:

        >>> class DummyOutput(_OutputFields):
        ...     def _validate_anomaly_map(self, anomaly_map):
        ...         return anomaly_map  # Implement actual validation
        ...     def _validate_pred_score(self, pred_score):
        ...         return pred_score  # Implement actual validation
        ...     def _validate_pred_mask(self, pred_mask):
        ...         return pred_mask  # Implement actual validation
        ...     def _validate_pred_label(self, pred_label):
        ...         return pred_label  # Implement actual validation

        >>> # Create an output instance with predictions
        >>> output = DummyOutput(
        ...     anomaly_map=torch.rand(224, 224),
        ...     pred_score=0.7,
        ...     pred_mask=torch.rand(224, 224) > 0.5,
        ...     pred_label=1
        ... )

        >>> # Access individual fields
        >>> anomaly_map = output.anomaly_map
        >>> score = output.pred_score
        >>> mask = output.pred_mask
        >>> label = output.pred_label

    Note:
        This is an abstract base class and is not intended to be instantiated
        directly. Concrete subclasses should implement all required validation
        methods.
    """

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
    """Mixin class for dataclasses that allows for in-place replacement of attributes.

    This mixin class provides a method for updating dataclass instances in place or
    by creating a new instance. It ensures that the updated instance is reinitialized
    by calling the ``__post_init__`` method if it exists.

    Examples:
        Assuming a dataclass `DummyItem` that uses UpdateMixin:

        >>> item = DummyItem(image=torch.rand(3, 224, 224), label=0)

        >>> # In-place update
        >>> item.update(label=1, pred_score=0.9)
        >>> print(item.label, item.pred_score)
        1 0.9

        >>> # Create a new instance with updates
        >>> new_item = item.update(in_place=False, image=torch.rand(3, 224, 224))
        >>> print(id(item) != id(new_item))
        True

        >>> # Update with multiple fields
        >>> item.update(label=2, pred_score=0.8, anomaly_map=torch.rand(224, 224))

    The `update` method can be used to modify single or multiple fields, either
    in-place or by creating a new instance. This flexibility is particularly useful
    in data processing pipelines and when working with model predictions in Anomalib.
    """

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
    """Generic dataclass for a single item in Anomalib datasets.

    This class combines input and output fields for anomaly detection tasks,
    providing a comprehensive representation of a single data item. It inherits
    from ``_InputFields`` for standard input data and ``_OutputFields`` for
    prediction results.

    The class also includes the ``UpdateMixin``, allowing for easy updates of
    field values. This is particularly useful during data processing pipelines
    and when working with model predictions.

    By using generic types, this class can accommodate various data types used
    in different Anomalib models and datasets, ensuring flexibility and
    reusability across the library.

    Examples:
        Assuming a concrete implementation ``DummyItem``:

        >>> class DummyItem(_GenericItem):
        ...     def _validate_image(self, image):
        ...         return image  # Implement actual validation
        ...     # Implement other required methods

        >>> # Create a generic item instance
        >>> item = DummyItem(
        ...     image=torch.rand(3, 224, 224),
        ...     gt_label=0,
        ...     pred_score=0.3,
        ...     anomaly_map=torch.rand(224, 224)
        ... )

        >>> # Access and update fields
        >>> image = item.image
        >>> item.update(pred_score=0.8, pred_label=1)

    Note:
        This is an abstract base class and is not intended to be instantiated
        directly. Concrete subclasses should implement all required validation
        methods.
    """


@dataclass
class _GenericBatch(
    UpdateMixin,
    Generic[T, ImageT, MaskT, PathT],
    _OutputFields[T, MaskT],
    _InputFields[T, ImageT, MaskT, PathT],
):
    """Generic dataclass for a batch of items in Anomalib datasets.

    This class represents a batch of data items, combining both input and output
    fields for anomaly detection tasks. It inherits from ``_InputFields`` for
    input data and ``_OutputFields`` for prediction results, allowing it to
    handle both training data and model outputs.

    The class includes the ``UpdateMixin``, enabling easy updates of field values
    across the entire batch. This is particularly useful for in-place modifications
    during data processing or when updating predictions.

    Examples:
        Assuming a concrete implementation ``DummyBatch``:

        >>> class DummyBatch(_GenericBatch):
        ...     def _validate_image(self, image):
        ...         return image  # Implement actual validation
        ...     # Implement other required methods

        >>> # Create a batch with input data
        >>> batch = DummyBatch(
        ...     image=torch.rand(32, 3, 224, 224),
        ...     gt_label=torch.randint(0, 2, (32,))
        ... )

        >>> # Update the entire batch with new predictions
        >>> batch.update(
        ...     pred_score=torch.rand(32),
        ...     anomaly_map=torch.rand(32, 224, 224)
        ... )

        >>> # Access individual fields
        >>> images = batch.image
        >>> labels = batch.gt_label
        >>> predictions = batch.pred_score

    Note:
        This is an abstract base class and is not intended to be instantiated
        directly. Concrete subclasses should implement all required validation
        methods.
    """


ItemT = TypeVar("ItemT", bound="_GenericItem")


@dataclass
class BatchIterateMixin(Generic[ItemT]):
    """Mixin class for iterating over batches of items in Anomalib datasets.

    This class provides functionality to iterate over individual items within a
    batch, convert batches to lists of items, and determine batch sizes. It's
    designed to work with Anomalib's batch processing pipelines.

    The mixin requires subclasses to define an ``item_class`` attribute, which
    specifies the class used for individual items in the batch. This ensures
    type consistency when iterating or converting batches.

    Key features include:
    - Iteration over batch items
    - Conversion of batches to lists of individual items
    - Batch size determination
    - A class method for collating individual items into a batch

    Examples:
        Assuming a subclass `DummyBatch` with `DummyItem` as its item_class:

        >>> batch = DummyBatch(images=[...], labels=[...])
        >>> for item in batch:
        ...     process_item(item)  # Iterate over items

        >>> item_list = batch.items  # Convert batch to list of items
        >>> type(item_list[0])
        <class 'DummyItem'>

        >>> batch_size = len(batch)  # Get batch size

        >>> items = [DummyItem(...) for _ in range(5)]
        >>> new_batch = DummyBatch.collate(items)  # Collate items into a batch

    This mixin enhances batch handling capabilities in Anomalib, facilitating
    efficient data processing and model interactions.
    """

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
