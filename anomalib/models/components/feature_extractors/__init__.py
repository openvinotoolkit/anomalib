"""Feature extractors."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .timm import TimmFeatureExtractor
from .torchfx import get_torchfx_feature_extractor

__all__ = ["TimmFeatureExtractor", "get_torchfx_feature_extractor"]
