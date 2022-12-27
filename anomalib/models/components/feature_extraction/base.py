"""Base feature extractor."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Dict, List, Tuple

import torch
from torch import nn


class BaseFeatureExtractor(nn.Module):
    """Base class for Feature Extractors."""

    def __init__(
        self,
    ):
        super().__init__()
        self.feature_extractor: nn.Module
        self.layers: List[str]

    @property
    @abstractmethod
    def out_dims(self) -> List[int]:
        """Returns the number of channels of the requested layers."""
        raise NotImplementedError("out_dims not implemented")

    def dryrun_find_featuremap_dims(self, input_shape: Tuple[int, int]) -> Dict[str, Dict]:
        """Dry run an empty image of get the feature map tensors' dimensions (num_features, resolution).

        Args:
            input_shape (Tuple[int, int]): Shape of the input image.

        Returns:
            Dict[str, Tuple]: mapping of ```layer -> dimensions dict```
                Each `dimension dict` has two keys: `num_features` (int) and ```resolution```(Tuple[int, int]).
        """

        dryrun_input = torch.empty(1, 3, *input_shape)
        dryrun_features = self.forward(dryrun_input)
        return {
            layer: {"num_features": dryrun_features[layer].shape[1], "resolution": dryrun_features[layer].shape[2:]}
            for layer in self.layers
        }
