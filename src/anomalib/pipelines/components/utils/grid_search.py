"""Utilities for grid search parameter iteration.

This module provides utilities for iterating over grid search parameter combinations
in a structured way. The main function :func:`get_iterator_from_grid_dict` takes a
dictionary of parameters and yields all possible combinations.

Example:
    >>> from anomalib.pipelines.components.utils import get_iterator_from_grid_dict
    >>> params = {
    ...     "model": {
    ...         "backbone": {"grid": ["resnet18", "resnet50"]},
    ...         "lr": {"grid": [0.001, 0.0001]}
    ...     }
    ... }
    >>> for config in get_iterator_from_grid_dict(params):
    ...     print(config)
    {'model': {'backbone': 'resnet18', 'lr': 0.001}}
    {'model': {'backbone': 'resnet18', 'lr': 0.0001}}
    {'model': {'backbone': 'resnet50', 'lr': 0.001}}
    {'model': {'backbone': 'resnet50', 'lr': 0.0001}}

The module handles:
    - Flattening nested parameter dictionaries
    - Generating all combinations of grid parameters
    - Reconstructing nested dictionary structure
    - Preserving non-grid parameters
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator
from itertools import product
from typing import Any

from anomalib.utils.config import convert_valuesview_to_tuple, flatten_dict, to_nested_dict


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
