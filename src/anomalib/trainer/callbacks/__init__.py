"""Necessary callbacks for Trainer."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .normalization import get_normalization_callback
from .post_processor import PostProcessorCallback

__all__ = ["PostProcessorCallback", "get_normalization_callback"]
