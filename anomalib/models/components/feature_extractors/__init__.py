"""Feature extractors."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from .timm import TimmFeatureExtractor
from .torchfx import TorchFXFeatureExtractor
from .utils import dryrun_find_featuremap_dims


class FeatureExtractor:
    """Compatibility wrapper for the old FeatureExtractor class.

    See :class:`anomalib.models.components.feature_extractors.timm.TimmFeatureExtractor` for more details.
    """

    def __init__(self, *args, **kwargs):
        logger = logging.getLogger(__name__)
        logger.warning("FeatureExtractor is deprecated. Use TimmFeatureExtractor instead.")
        self.feature_extractor = TimmFeatureExtractor(*args, **kwargs)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Calls the TimmFeatureExtractor."""
        return self.feature_extractor(*args, **kwds)


__all__ = ["TimmFeatureExtractor", "TorchFXFeatureExtractor", "dryrun_find_featuremap_dims", "FeatureExtractor"]
