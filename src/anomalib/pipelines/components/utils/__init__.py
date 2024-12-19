"""Utility functions for pipeline components.

This module provides utility functions used by various pipeline components for tasks
like:

- Grid search parameter iteration via :func:`get_iterator_from_grid_dict`
- Other utility functions for pipeline execution

Example:
    >>> from anomalib.pipelines.components.utils import get_iterator_from_grid_dict
    >>> params = {"lr": [0.1, 0.01], "batch_size": [32, 64]}
    >>> iterator = get_iterator_from_grid_dict(params)
    >>> for config in iterator:
    ...     print(config)
    {"lr": 0.1, "batch_size": 32}
    {"lr": 0.1, "batch_size": 64}
    {"lr": 0.01, "batch_size": 32}
    {"lr": 0.01, "batch_size": 64}
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .grid_search import get_iterator_from_grid_dict

__all__ = ["get_iterator_from_grid_dict"]
