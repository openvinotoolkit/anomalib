"""Tools for anomaly score normalization.

This module provides utilities for normalizing anomaly scores in anomaly detection
tasks. The utilities include:
    - Min-max normalization to scale scores to [0,1] range
    - Enum class to specify normalization methods

Example:
    >>> from anomalib.utils.normalization import NormalizationMethod
    >>> # Use min-max normalization
    >>> method = NormalizationMethod.MIN_MAX
    >>> print(method)
    min_max
    >>> # Use no normalization
    >>> method = NormalizationMethod.NONE
    >>> print(method)
    none

The module ensures consistent normalization of anomaly scores across different
detection algorithms.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class NormalizationMethod(str, Enum):
    """Enumeration of supported normalization methods for anomaly scores.

    This enum class defines the available methods for normalizing anomaly scores:
        - ``MIN_MAX``: Scales scores to [0,1] range using min-max normalization
        - ``NONE``: No normalization is applied, raw scores are used

    Example:
        >>> from anomalib.utils.normalization import NormalizationMethod
        >>> # Use min-max normalization
        >>> method = NormalizationMethod.MIN_MAX
        >>> print(method)
        min_max
        >>> # Use no normalization
        >>> method = NormalizationMethod.NONE
        >>> print(method)
        none

    The enum inherits from ``str`` to enable string comparison and serialization.
    """

    MIN_MAX = "min_max"
    NONE = "none"
