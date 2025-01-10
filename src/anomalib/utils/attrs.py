"""Utility functions for working with attributes."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any


def get_nested_attr(obj: Any, attr_path: str, default: Any | None = None) -> Any:  # noqa: ANN401
    """Safely retrieves a nested attribute from an object.

    This function helps reduce boilerplate code when working with nested attributes, by allowing you to retrieve a
    nested attribute with a single function call instead of multiple nested calls to `getattr`.

    Args:
        obj: The object to retrieve the attribute from.
        attr_path: A dot-separated string representing the attribute path.
        default: The default value to return if any attribute in the path is missing.

    Returns:
        The value of the nested attribute, or `default` if any attribute in the path is missing.

    Example:
        >>> class A:
        ...     def __init__(self, b):
        ...         self.b = b
        >>>
        >>> class B:
        ...     def __init__(self, c):
        ...         self.c = c
        >>>
        >>> class C:
        ...     def __init__(self, d):
        ...         self.d = d
        >>>
        >>> d = 42
        >>> c = C(d)
        >>> b = B(c)
        >>> a = A(b)
        >>> get_nested_attr(a, "b.c.d")  # 42
        >>> # this is equivalent to:
        >>> # getattr(getattr(getattr(a, "b", None), "c", None), "value", None)
        >>>
        >>> get_nested_attr(a, "b.c.foo")  # None
        >>> get_nested_attr(a, "b.c.foo", "bar")  # "bar"
        >>> get_nested_attr(a, "b.d.c")  # None
    """
    for attr in attr_path.split("."):
        obj = getattr(obj, attr, default)
        if obj is default:
            return default
    return obj
