"""Utility functions for working with Torchvision Transforms."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from torchvision.transforms.v2 import Compose, Transform


def extract_transforms_by_type(
    input_transform: Transform | None,
    transform_type: type[Transform],
) -> list[type[Transform]]:
    """Extracts all transforms of a given type from a transform or transform composition.

    Args:
        input_transform (Transform): Torchvision Transform instance.
        transform_type (Type[Transform]): Type of transform to retrieve.

    Returns:
        List[Transform]: List of Resize transform instances.
    """
    if isinstance(input_transform, transform_type):
        return [input_transform]
    if isinstance(input_transform, Compose):
        return [transform for transform in input_transform.transforms if isinstance(transform, transform_type)]
    return []
