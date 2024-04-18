"""Utils for benchmarking."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
from collections.abc import Generator
from itertools import product
from typing import Any

from jsonargparse import Namespace
from jsonargparse._typehints import ActionTypeHint

from anomalib.pipelines.utils import (
    convert_to_tuple,
    dict_from_namespace,
    flatten_dict,
    namespace_from_dict,
    to_nested_dict,
)


def get_iterator_from_grid_dict(container: dict) -> Generator[dict, Any, None]:
    """Yields an iterator based on the grid search arguments.

    Args:
        container (dict): Container with grid search arguments.

    Example:
        >>> container = {
                "seed": 42,
                "data": {
                    "root": ...,
                    "category": {
                        "grid": ["bottle", "carpet"],
                        ...
                    }
                }
            }
        >>> get_iterator_from_grid_search(container)
        {
                "seed": 42,
                "data": {
                    "root": ...,
                    "category": "bottle"
                        ...
                    }
                }
        }

    Yields:
        Generator[dict, Any, None]: Iterator based on the grid search arguments.
    """
    _container = flatten_dict(container)
    grid_dict = {key: value for key, value in _container.items() if "grid" in key}
    _container = {key: value for key, value in _container.items() if key not in grid_dict}
    combinations = list(product(*convert_to_tuple(grid_dict.values())))
    for combination in combinations:
        for key, value in zip(grid_dict.keys(), combination, strict=True):
            _container[key.removesuffix(".grid")] = value
        yield to_nested_dict(_container)


class GridSearchAction(ActionTypeHint):
    """Grid search action for jsonargparse.

    Allows using grid search key in the arguments.

    Example:
        ```yaml

            nested_key:
                class_path: str
                init_args: [arg1, arg2]
        ```
        or
        ```yaml
            nested_key:
                class_path:
                    grid: [val1, val2]
                init_args:
                    - arg1
                        grid: [val1, val2]
                    - arg2
                        ...
        ```

    """

    def __call__(self, *args, **kwargs) -> "GridSearchAction | None":
        """Parse arguments for grid search."""
        if len(args) == 0:
            kwargs["_typehint"] = self._typehint
            kwargs["_enable_path"] = self._enable_path
            return GridSearchAction(**kwargs)
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
                    GridSearchAction._crawl(val, parents, parent_key)

    @staticmethod
    def _pop_nested_key(container: dict, key: str) -> None:
        """Pop the nested key from the container."""
        keys = key.split(".")
        if len(keys) > 1:
            GridSearchAction._pop_nested_key(container[keys[0]], ".".join(keys[1:]))
        elif key != "":
            container.pop(keys[0])
        else:
            # it is possible that the container is {"grid": [...]}
            container.clear()

    @staticmethod
    def sanitize_value(value: dict | Namespace) -> dict:
        """Returns a new value with all grid search keys removed."""
        _value = dict_from_namespace(value) if isinstance(value, Namespace) else copy.deepcopy(value)
        keys: list[str] = []
        GridSearchAction._crawl(_value, keys)
        for key in keys:
            GridSearchAction._pop_nested_key(_value, key)
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
        _value = GridSearchAction.sanitize_value(value)
        # check only the keys for which grid is not assigned.
        # this ensures that single keys are checked against the class
        _value = namespace_from_dict(_value)
        if vars(_value):
            super()._check_type(_value, append, cfg)
        # convert original value to Namespace recursively
        value = namespace_from_dict(value)
        GridSearchAction.discard_init_args_on_class_path_change(self, value, _value)
        _value.update(value)
        GridSearchAction.apply_appends(self, _value)

        return _value
