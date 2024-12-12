"""Utility functions for transforms."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from torchvision.transforms.v2 import CenterCrop, Compose, Resize, Transform

from anomalib.data.transforms import ExportableCenterCrop


def get_exportable_transform(transform: Transform | None) -> Transform | None:
    """Get exportable transform.

    Some transforms are not supported by ONNX/OpenVINO, so we need to replace them with exportable versions.
    """
    if transform is None:
        return None
    transform = disable_antialiasing(transform)
    return convert_center_crop_transform(transform)


def disable_antialiasing(transform: Transform) -> Transform:
    """Disable antialiasing in Resize transforms.

    Resizing with antialiasing is not supported by ONNX, so we need to disable it.
    """
    if isinstance(transform, Resize):
        transform.antialias = False
    if isinstance(transform, Compose):
        for tr in transform.transforms:
            disable_antialiasing(tr)
    return transform


def convert_center_crop_transform(transform: Transform) -> Transform:
    """Convert CenterCrop to ExportableCenterCrop.

    Torchvision's CenterCrop is not supported by ONNX, so we need to replace it with our own ExportableCenterCrop.
    """
    if isinstance(transform, CenterCrop):
        transform = ExportableCenterCrop(size=transform.size)
    if isinstance(transform, Compose):
        for index in range(len(transform.transforms)):
            tr = transform.transforms[index]
            transform.transforms[index] = convert_center_crop_transform(tr)
    return transform
