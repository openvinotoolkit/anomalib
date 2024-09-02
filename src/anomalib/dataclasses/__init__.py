"""Anomalib dataclasses."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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
