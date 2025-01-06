"""Utility functions for transforms.

This module provides utility functions for managing transforms in the pre-processing
pipeline.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy

from torchvision.transforms.v2 import CenterCrop, Compose, Resize, Transform

from anomalib.data.transforms import ExportableCenterCrop


def get_exportable_transform(transform: Transform | None) -> Transform | None:
    """Get an exportable version of a transform.

    This function converts a torchvision transform into a format that is compatible with
    ONNX and OpenVINO export. It handles two main compatibility issues:

    1. Disables antialiasing in ``Resize`` transforms
    2. Converts ``CenterCrop`` to ``ExportableCenterCrop``

    Args:
        transform (Transform | None): The transform to convert. If ``None``, returns
            ``None``.

    Returns:
        Transform | None: The converted transform that is compatible with ONNX/OpenVINO
            export. Returns ``None`` if input transform is ``None``.

    Example:
        >>> from torchvision.transforms.v2 import Compose, Resize, CenterCrop
        >>> transform = Compose([
        ...     Resize((224, 224), antialias=True),
        ...     CenterCrop(200)
        ... ])
        >>> exportable = get_exportable_transform(transform)
        >>> # Now transform is compatible with ONNX/OpenVINO export

    Note:
        Some torchvision transforms are not directly supported by ONNX/OpenVINO. This
        function handles the most common cases, but additional transforms may need
        special handling.
    """
    if transform is None:
        return None
    transform = copy.deepcopy(transform)
    transform = disable_antialiasing(transform)
    return convert_center_crop_transform(transform)


def disable_antialiasing(transform: Transform) -> Transform:
    """Disable antialiasing in Resize transforms.

    This function recursively disables antialiasing in any ``Resize`` transforms found
    within the provided transform or transform composition. This is necessary because
    antialiasing is not supported during ONNX export.

    Args:
        transform (Transform): Transform or composition of transforms to process.

    Returns:
        Transform: The processed transform with antialiasing disabled in any
            ``Resize`` transforms.

    Example:
        >>> from torchvision.transforms.v2 import Compose, Resize
        >>> transform = Compose([
        ...     Resize((224, 224), antialias=True),
        ...     Resize((256, 256), antialias=True)
        ... ])
        >>> transform = disable_antialiasing(transform)
        >>> # Now all Resize transforms have antialias=False

    Note:
        This function modifies the transforms in-place by setting their
        ``antialias`` attribute to ``False``. The original transform object is
        returned.
    """
    if isinstance(transform, Resize):
        transform.antialias = False
    if isinstance(transform, Compose):
        for tr in transform.transforms:
            disable_antialiasing(tr)
    return transform


def convert_center_crop_transform(transform: Transform) -> Transform:
    """Convert torchvision's CenterCrop to ExportableCenterCrop.

    This function recursively converts any ``CenterCrop`` transforms found within the
    provided transform or transform composition to ``ExportableCenterCrop``. This is
    necessary because torchvision's ``CenterCrop`` is not supported during ONNX
    export.

    Args:
        transform (Transform): Transform or composition of transforms to process.

    Returns:
        Transform: The processed transform with all ``CenterCrop`` transforms
            converted to ``ExportableCenterCrop``.

    Example:
        >>> from torchvision.transforms.v2 import Compose, CenterCrop
        >>> transform = Compose([
        ...     CenterCrop(224),
        ...     CenterCrop((256, 256))
        ... ])
        >>> transform = convert_center_crop_transform(transform)
        >>> # Now all CenterCrop transforms are converted to ExportableCenterCrop

    Note:
        This function creates new ``ExportableCenterCrop`` instances to replace the
        original ``CenterCrop`` transforms. The original transform object is
        returned with the replacements applied.
    """
    if isinstance(transform, CenterCrop):
        transform = ExportableCenterCrop(size=transform.size)
    if isinstance(transform, Compose):
        for index in range(len(transform.transforms)):
            tr = transform.transforms[index]
            transform.transforms[index] = convert_center_crop_transform(tr)
    return transform
