"""Anomalib dataclasses."""

from .numpy import (
    NumpyImageInputBatch,
    NumpyImageInputItem,
    NumpyImageOutputBatch,
    NumpyImageOutputItem,
    NumpyVideoInputBatch,
    NumpyVideoInputItem,
    NumpyVideoOutputBatch,
    NumpyVideoOutputItem,
)
from .torch import (
    InferenceBatch,
    TorchImageInputBatch,
    TorchImageInputItem,
    TorchImageOutputBatch,
    TorchImageOutputItem,
    TorchVideoInputBatch,
    TorchVideoInputItem,
    TorchVideoOutputBatch,
    TorchVideoOutputItem,
)

__all__ = [
    "InferenceBatch",
    "TorchImageInputItem",
    "TorchImageInputBatch",
    "TorchImageOutputItem",
    "TorchImageOutputBatch",
    "TorchVideoInputItem",
    "TorchVideoInputBatch",
    "TorchVideoOutputItem",
    "TorchVideoOutputBatch",
    "NumpyImageInputItem",
    "NumpyImageInputBatch",
    "NumpyImageOutputItem",
    "NumpyImageOutputBatch",
    "NumpyVideoInputItem",
    "NumpyVideoInputBatch",
    "NumpyVideoOutputItem",
    "NumpyVideoOutputBatch",
]
