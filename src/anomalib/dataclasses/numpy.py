"""Dataclasses for numpy data."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Generic

import numpy as np

from .generic import BatchMixin, PathT, _GenericItem, _InputFields, _OutputFields, _VideoInputFields


# torch inputs
@dataclass
class _NumpyInput(
    Generic[PathT],
    _InputFields[np.ndarray, np.ndarray, np.ndarray, PathT],
):
    """Dataclass for numpy image input.

    Common class for both single item and batch.
    """


@dataclass
class _NumpyInputItem(
    _NumpyInput[Path],
    _GenericItem[np.ndarray, np.ndarray, np.ndarray, Path],
):
    """Dataclass for numpy input item.

    Common class for any modality (image, video, etc.).
    """


@dataclass
class _NumpyInputBatch(
    _NumpyInput[list[Path]],
):
    """Dataclass for numpy input batch.

    Common class for any modality (image, video, etc.).
    """


# torch image inputs
@dataclass
class NumpyImageInputItem(
    _NumpyInputItem,
):
    """Dataclass for numpy image input item."""


@dataclass
class NumpyImageInputBatch(
    BatchMixin,
    _NumpyInputBatch,
):
    """Dataclass for numpy image input batch."""

    @property
    def item_class(self) -> Callable:
        """Get the single item equivalent of the batch class."""
        return NumpyImageInputItem


# torch video inputs
@dataclass
class _NumpyVideoInput(
    Generic[PathT],
    _VideoInputFields[np.ndarray, np.ndarray, np.ndarray, PathT],
):
    """Dataclass for numpy video input.

    Common class for both single item and batch.
    """


@dataclass
class NumpyVideoInputItem(
    _NumpyInputItem,
    _NumpyVideoInput[Path],
):
    """Dataclass for numpy video input item."""


@dataclass
class NumpyVideoInputBatch(
    BatchMixin,
    _NumpyInputBatch,
    _NumpyVideoInput[list[Path]],
):
    """Dataclass for numpy video input batch."""

    @property
    def item_class(self) -> Callable:
        """Get the single item equivalent of the batch class."""
        return NumpyVideoInputItem


###### OUTPUTS ######


# torch outputs
@dataclass
class _NumpyOutput(
    Generic[PathT],
    _OutputFields[np.ndarray, np.ndarray],
    _NumpyInput[PathT],
):
    """Dataclass for numpy output.

    Common class for both single item and batch, and any modality (image, video, etc.).
    """


@dataclass
class _NumpyOutputItem(
    _NumpyOutput[Path],
):
    """Dataclass for numpy output item.

    Common class for any modality (image, video, etc.).
    """


@dataclass
class _NumpyOutputBatch(
    _NumpyOutput[list[Path]],
    _NumpyInputBatch,
):
    """Dataclass for numpy output batch.

    Common class for any modality (image, video, etc.).
    """


# torch image outputs
@dataclass
class NumpyImageOutputItem(
    _NumpyOutputItem,
):
    """Dataclass for numpy image output item."""


@dataclass
class NumpyImageOutputBatch(
    BatchMixin,
    _NumpyOutputBatch,
):
    """Dataclass for numpy image output batch."""

    @property
    def item_class(self) -> Callable:
        """Get the single item equivalent of the batch class."""
        return NumpyImageOutputItem


# torch video outputs
@dataclass
class NumpyVideoOutputItem(
    _NumpyOutputItem,
    _NumpyVideoInput[Path],
):
    """Dataclass for numpy video output item."""


@dataclass
class NumpyVideoOutputBatch(
    BatchMixin,
    _NumpyOutputBatch,
    _NumpyVideoInput[list[Path]],
):
    """Dataclass for numpy video output batch."""

    @property
    def item_class(self) -> Callable:
        """Get the single item equivalent of the batch class."""
        return NumpyVideoOutputItem
