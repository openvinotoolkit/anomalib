"""Anomalib dataclasses."""

from .numpy import (
    NumpyImageBatch,
    NumpyImageItem,
    NumpyVideoBatch,
    NumpyVideoItem,
)
from .torch import (
    Batch,
    DepthBatch,
    DepthItem,
    ImageBatch,
    ImageItem,
    InferenceBatch,
    Item,
    VideoBatch,
    VideoItem,
)

__all__ = [
    "Item",
    "Batch",
    "InferenceBatch",
    "ImageItem",
    "ImageBatch",
    "VideoItem",
    "VideoBatch",
    "NumpyImageItem",
    "NumpyImageBatch",
    "NumpyVideoItem",
    "NumpyVideoBatch",
    "DepthItem",
    "DepthBatch",
]
