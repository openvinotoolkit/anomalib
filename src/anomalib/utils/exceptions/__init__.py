"""Exception and error handling utilities for anomaly detection.

This module provides utilities for handling exceptions and errors in the anomalib
library. The utilities include:
    - Dynamic import handling with graceful fallbacks
    - Custom exception types for anomaly detection
    - Error handling helpers and decorators

Example:
    >>> from anomalib.utils.exceptions import try_import
    >>> # Try importing an optional dependency
    >>> torch_fidelity = try_import("torch_fidelity")
    >>> if torch_fidelity is None:
    ...     print("torch-fidelity not installed")

The module ensures consistent and informative error handling across the anomalib
codebase.
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .imports import try_import

__all__ = ["try_import"]
