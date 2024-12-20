"""Utility functions for working with attributes."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any


def get_nested_attr(obj: Any, attr_path: str, default: Any | None = None) -> Any:  # noqa: ANN401
    """Safely retrieves a nested attribute from an object.

    Args:
        obj: The object to retrieve the attribute from.
        attr_path: A dot-separated string representing the attribute path.
        default: The default value to return if any attribute in the path is missing.

    Returns:
        The value of the nested attribute, or `default` if any attribute in the path is missing.
    """
    for attr in attr_path.split("."):
        obj = getattr(obj, attr, default)
        if obj is default:
            return default
    return obj
