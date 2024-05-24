"""Utils for benchmarking."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator
from itertools import product
from typing import Any

from anomalib.utils.config import (
    convert_valuesview_to_tuple,
    flatten_dict,
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
    combinations = list(product(*convert_valuesview_to_tuple(grid_dict.values())))
    for combination in combinations:
        for key, value in zip(grid_dict.keys(), combination, strict=True):
            _container[key.removesuffix(".grid")] = value
        yield to_nested_dict(_container)
