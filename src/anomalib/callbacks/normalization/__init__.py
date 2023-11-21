"""Normalization callbacks.

Note: These callbacks are used within the Engine.
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from .cdf_normalization import _CdfNormalizationCallback
from .min_max_normalization import _MinMaxNormalizationCallback
from .utils import get_normalization_callback

__all__ = ["get_normalization_callback", "_CdfNormalizationCallback", "_MinMaxNormalizationCallback"]
