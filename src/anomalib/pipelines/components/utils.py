"""Utility functions."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import functools
import io
import sys
from collections.abc import Callable, Iterable, ValuesView
from typing import Any

from jsonargparse import Namespace


def hide_output(func: Callable[..., Any]) -> Callable[..., Any]:
    """Hide output of the function.

    Args:
        func (function): Hides output of this function.

    Raises:
        Exception: In case the execution of function fails, it raises an exception.

    Returns:
        object of the called function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:  # noqa: ANN401
        std_out = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            value = func(*args, **kwargs)
        # NOTE: A generic exception is used here to catch all exceptions.
        except Exception as exception:  # noqa: BLE001
            raise Exception(buf.getvalue()) from exception  # noqa: TRY002
        sys.stdout = std_out
        return value

    return wrapper


def dict_from_namespace(container: Namespace) -> dict:
    """Convert Namespace to dictionary recursively."""
    output = {}
    for k, v in container.__dict__.items():
        if isinstance(v, Namespace):
            output[k] = dict_from_namespace(v)
        else:
            output[k] = v
    return output


def namespace_from_dict(container: dict) -> Namespace:
    """Convert dictionary to Namespace recursively."""
    output = Namespace()
    for k, v in container.items():
        if isinstance(v, dict):
            setattr(output, k, namespace_from_dict(v))
        else:
            setattr(output, k, v)
    return output


def flatten_dict(config: dict, prefix: str = "") -> dict:
    """Flatten the dictionary."""
    out = {}
    for key, value in config.items():
        if isinstance(value, dict):
            out.update(flatten_dict(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def convert_to_tuple(values: ValuesView) -> list[tuple]:
    """Convert a ValuesView object to a list of tuples.

    This is useful to get list of possible values for each parameter in the config and a tuple for values that are
    are to be patched. Ideally this is useful when used with product.

    Example:
        >>> params = DictConfig({
                "dataset.category": [
                    "bottle",
                    "cable",
                ],
                "dataset.image_size": 224,
                "model_name": ["padim"],
            })
        >>> convert_to_tuple(params.values())
        [('bottle', 'cable'), (224,), ('padim',)]
        >>> list(itertools.product(*convert_to_tuple(params.values())))
        [('bottle', 224, 'padim'), ('cable', 224, 'padim')]

    Args:
        values: ValuesView: ValuesView object to be converted to a list of tuples.

    Returns:
        list[Tuple]: List of tuples.
    """
    return_list = []
    for value in values:
        if isinstance(value, Iterable) and not isinstance(value, str):
            return_list.append(tuple(value))
        else:
            return_list.append((value,))
    return return_list


def to_nested_dict(config: dict) -> dict:
    """Convert the flattened dictionary to nested dictionary."""
    out: dict[str, Any] = {}
    for key, value in config.items():
        keys = key.split(".")
        _dict = out
        for k in keys[:-1]:
            _dict = _dict.setdefault(k, {})
        _dict[keys[-1]] = value
    return out
