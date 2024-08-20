"""Anomalib dataclasses."""

from .numpy import (
    NumpyImageBatch,
    NumpyImageItem,
    NumpyVideoBatch,
    NumpyVideoItem,
)
from .torch import (
    ImageBatch,
    ImageItem,
    InferenceBatch,
    VideoBatch,
    VideoItem,
)

__all__ = [
    "InferenceBatch",
    "ImageItem",
    "ImageBatch",
    "VideoItem",
    "VideoBatch",
    "NumpyImageItem",
    "NumpyImageBatch",
    "NumpyVideoItem",
    "NumpyVideoBatch",
]
