"""Dataclasses for numpy data."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .generic import _GenericBatch, _GenericItem, _InputFields, _OutputFields, _VideoInputFields


# torch image outputs
@dataclass
class NumpyImageOutputItem(
    _GenericItem,
    _OutputFields[np.ndarray, np.ndarray],
    _InputFields[np.ndarray, np.ndarray, np.ndarray, Path],
):
    """Dataclass for numpy image output item."""


@dataclass
class NumpyImageOutputBatch(
    _GenericBatch[NumpyImageOutputItem],
    _OutputFields[np.ndarray, np.ndarray],
    _InputFields[np.ndarray, np.ndarray, np.ndarray, list[Path]],
):
    """Dataclass for numpy image output batch."""

    @property
    def item_class(self) -> Callable:
        """Get the single item equivalent of the batch class."""
        return NumpyImageOutputItem


# torch video outputs
@dataclass
class NumpyVideoOutputItem(
    _GenericItem,
    _OutputFields[np.ndarray, np.ndarray],
    _VideoInputFields[np.ndarray, np.ndarray, np.ndarray, Path],
):
    """Dataclass for numpy video output item."""


@dataclass
class NumpyVideoOutputBatch(
    _GenericBatch[NumpyVideoOutputItem],
    _OutputFields[np.ndarray, np.ndarray],
    _VideoInputFields[np.ndarray, np.ndarray, np.ndarray, list[Path]],
):
    """Dataclass for numpy video output batch."""

    @property
    def item_class(self) -> Callable:
        """Get the single item equivalent of the batch class."""
        return NumpyVideoOutputItem
