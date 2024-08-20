"""Dataclasses for numpy data."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .generic import _GenericBatch, _GenericItem, _InputFields, _OutputFields, _VideoInputFields


# torch image outputs
@dataclass
class NumpyImageItem(
    _GenericItem,
    _OutputFields[np.ndarray, np.ndarray],
    _InputFields[np.ndarray, np.ndarray, np.ndarray, Path],
):
    """Dataclass for numpy image output item."""


@dataclass
class NumpyImageBatch(
    _GenericBatch[NumpyImageItem],
    _OutputFields[np.ndarray, np.ndarray],
    _InputFields[np.ndarray, np.ndarray, np.ndarray, list[Path]],
):
    """Dataclass for numpy image output batch."""

    item_class = NumpyImageItem


# torch video outputs
@dataclass
class NumpyVideoItem(
    _GenericItem,
    _OutputFields[np.ndarray, np.ndarray],
    _VideoInputFields[np.ndarray, np.ndarray, np.ndarray, Path],
):
    """Dataclass for numpy video output item."""


@dataclass
class NumpyVideoBatch(
    _GenericBatch[NumpyVideoItem],
    _OutputFields[np.ndarray, np.ndarray],
    _VideoInputFields[np.ndarray, np.ndarray, np.ndarray, list[Path]],
):
    """Dataclass for numpy video output batch."""

    item_class = NumpyVideoItem
