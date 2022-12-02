"""Feature extractors."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from .timm import TimmFeatureExtractor, TimmFeatureExtractorParams
from .torchfx import TorchFXFeatureExtractor, TorchFXFeatureExtractorParams
from .wrapper import FeatureExtractorParams, get_feature_extractor

__all__ = [
    "get_feature_extractor",
    "FeatureExtractorParams",
    "TimmFeatureExtractor",
    "TimmFeatureExtractorParams",
    "TorchFXFeatureExtractor",
    "TorchFXFeatureExtractorParams",
]
