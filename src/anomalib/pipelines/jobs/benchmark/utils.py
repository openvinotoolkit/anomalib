"""Utils for benchmarking."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
from collections.abc import Iterable, ValuesView
from typing import Any

from jsonargparse import Namespace
from jsonargparse._typehints import ActionTypeHint


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


class _GridSearchAction(ActionTypeHint):
    def __call__(self, *args, **kwargs) -> "_GridSearchAction | None":
        """Parse arguments for grid search."""
        if len(args) == 0:
            kwargs["_typehint"] = self._typehint
            kwargs["_enable_path"] = self._enable_path
            return _GridSearchAction(**kwargs)
        return None

    @staticmethod
    def _crawl(value: Any, parents: list[str], parent: str = "") -> None:  # noqa: ANN401
        """Crawl through the dictionary and store path to parents."""
        if isinstance(value, dict):
            for key, val in value.items():
                if key == "grid":
                    parents.append(parent)
                elif isinstance(val, dict):
                    parent_key = f"{parent}.{key}" if parent else key
                    _GridSearchAction._crawl(val, parents, parent_key)

    @staticmethod
    def pop_nested_key(container: dict, key: str) -> None:
        keys = key.split(".")
        if len(keys) > 1:
            _GridSearchAction.pop_nested_key(container[keys[0]], ".".join(keys[1:]))
        else:
            container.pop(keys[0])

    @staticmethod
    def sanitize_value(value: dict | Namespace) -> dict:
        """Returns a new value with all grid search keys removed."""
        _value = dict_from_namespace(value) if isinstance(value, Namespace) else copy.deepcopy(value)
        keys: list[str] = []
        _GridSearchAction._crawl(_value, keys)
        for key in keys:
            _GridSearchAction.pop_nested_key(_value, key)
        return _value

    def _check_type(self, value: dict | Namespace | Any, append: bool = False, cfg: Namespace | None = None) -> Any:  # noqa: ANN401
        """Ignore all grid search keys.

        This allows the args to follow the same format as ``add_subclass_arguments``
        nested_key:
            class_path: str
            init_args: [arg1, arg2]
        at the same time allows grid search key
        nested_key:
            class_path:
                grid: [val1, val2]
            init_args:
                - arg1
                    grid: [val1, val2]
                - arg2
                    ...
        """
        if not isinstance(value, dict | Namespace):
            return value
        _value = _GridSearchAction.sanitize_value(value)
        # check only the keys for which grid is not assigned.
        # this ensures that single keys are checked against the class
        _value = namespace_from_dict(_value)
        if vars(_value):
            super()._check_type(_value, append, cfg)
        # convert original value to Namespace recursively
        value = namespace_from_dict(value)
        _GridSearchAction.discard_init_args_on_class_path_change(self, value, _value)
        _value.update(value)
        _GridSearchAction.apply_appends(self, _value)

        return _value
