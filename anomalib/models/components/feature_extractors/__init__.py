"""Feature extractors."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from typing import Union

from torch import Tensor, nn

from .timm import TimmFeatureExtractor, TimmFeatureExtractorParams
from .torchfx import TorchFXBackboneParams, TorchFXFeatureExtractor
from .utils import dryrun_find_featuremap_dims


class FeatureExtractor(nn.Module):
    """Selects either timm or torchfx feature extractor based on the arguments passed.

    If you want to use timm feature extractor, you need to pass the following arguments:
    backbone,layers,pre_trained,requires_grad

    If you want to use torchfx feature extractor, you need to pass the following arguments:
    backbone,return_nodes,weights,requires_grad
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.feature_extractor: Union[TorchFXFeatureExtractor, TimmFeatureExtractor]
        self._check_type(**kwargs)

    def _check_type(self, **kwargs):
        if "pre_trained" in kwargs:
            self.feature_extractor = TimmFeatureExtractor(**kwargs)
        else:
            self.feature_extractor = TorchFXFeatureExtractor(**kwargs)

    def forward(self, inputs: Tensor) -> Tensor:
        """Returns outputs of the feature extractor."""
        return self.feature_extractor(inputs)


__all__ = [
    "dryrun_find_featuremap_dims",
    "FeatureExtractor",
    "TimmFeatureExtractor",
    "TimmFeatureExtractorParams",
    "TorchFXBackboneParams",
    "TorchFXFeatureExtractor",
]
