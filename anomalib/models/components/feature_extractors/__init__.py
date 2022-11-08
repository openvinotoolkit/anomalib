"""Feature extractors."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .feature_extractor import FeatureExtractor
from .utils import dryrun_find_featuremap_dims

__all__ = ["FeatureExtractor", "dryrun_find_featuremap_dims"]
