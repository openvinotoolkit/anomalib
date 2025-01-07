"""Helper utilities for data.

This module provides various utility functions for data handling in Anomalib.

The utilities are organized into several categories:

- Image handling: Functions for reading, writing and processing images
- Box handling: Functions for converting between masks and bounding boxes
- Path handling: Functions for validating and resolving file paths
- Dataset splitting: Functions for splitting datasets into train/val/test
- Data generation: Functions for generating synthetic data like Perlin noise
- Download utilities: Functions for downloading and extracting datasets

Example:
    >>> from anomalib.data.utils import read_image, generate_perlin_noise
    >>> # Read an image
    >>> image = read_image("path/to/image.jpg")
    >>> # Generate Perlin noise
    >>> noise = generate_perlin_noise(256, 256)
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .boxes import boxes_to_anomaly_maps, boxes_to_masks, masks_to_boxes
from .download import DownloadInfo, download_and_extract
from .generators import generate_perlin_noise
from .image import (
    generate_output_image_filename,
    get_image_filenames,
    get_image_height_and_width,
    read_depth_image,
    read_image,
    read_mask,
)
from .label import LabelName
from .path import (
    DirType,
    _check_and_convert_path,
    _prepare_files_labels,
    resolve_path,
    validate_and_resolve_path,
    validate_path,
)
from .split import Split, TestSplitMode, ValSplitMode, concatenate_datasets, random_split, split_by_label

__all__ = [
    "generate_output_image_filename",
    "get_image_filenames",
    "get_image_height_and_width",
    "generate_perlin_noise",
    "read_image",
    "read_mask",
    "read_depth_image",
    "random_split",
    "split_by_label",
    "concatenate_datasets",
    "Split",
    "ValSplitMode",
    "TestSplitMode",
    "LabelName",
    "DirType",
    "masks_to_boxes",
    "boxes_to_masks",
    "boxes_to_anomaly_maps",
    "download_and_extract",
    "DownloadInfo",
    "_check_and_convert_path",
    "_prepare_files_labels",
    "resolve_path",
    "validate_path",
    "validate_and_resolve_path",
]
