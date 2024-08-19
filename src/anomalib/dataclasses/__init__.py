"""Anomalib dataclasses."""

from .numpy import (
    NumpyImageOutputBatch,
    NumpyImageOutputItem,
    NumpyVideoOutputBatch,
    NumpyVideoOutputItem,
)
from .torch import (
    InferenceBatch,
    TorchImageOutputBatch,
    TorchImageOutputItem,
    TorchVideoOutputBatch,
    TorchVideoOutputItem,
)

__all__ = [
    "InferenceBatch",
    "TorchImageOutputItem",
    "TorchImageOutputBatch",
    "TorchVideoOutputItem",
    "TorchVideoOutputBatch",
    "NumpyImageOutputItem",
    "NumpyImageOutputBatch",
    "NumpyVideoOutputItem",
    "NumpyVideoOutputBatch",
]
