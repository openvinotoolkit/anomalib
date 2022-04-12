"""Anomaly Map Generator for CFlow model implementation."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import List, Tuple, Union, cast

import torch
import torch.nn.functional as F
from omegaconf import ListConfig
from torch import Tensor


class AnomalyMapGenerator:
    """Generate Anomaly Heatmap."""

    def __init__(
        self,
        image_size: Union[ListConfig, Tuple],
        pool_layers: List[str],
    ):
        self.distance = torch.nn.PairwiseDistance(p=2, keepdim=True)
        self.image_size = image_size if isinstance(image_size, tuple) else tuple(image_size)
        self.pool_layers: List[str] = pool_layers

    def compute_anomaly_map(
        self, distribution: Union[List[Tensor], List[List]], height: List[int], width: List[int]
    ) -> Tensor:
        """Compute the layer map based on likelihood estimation.

        Args:
          distribution: Probability distribution for each decoder block
          height: blocks height
          width: blocks width

        Returns:
          Final Anomaly Map

        """

        test_map: List[Tensor] = []
        for layer_idx in range(len(self.pool_layers)):
            test_norm = torch.tensor(distribution[layer_idx], dtype=torch.double)  # pylint: disable=not-callable
            test_norm -= torch.max(test_norm)  # normalize likelihoods to (-Inf:0] by subtracting a constant
            test_prob = torch.exp(test_norm)  # convert to probs in range [0:1]
            test_mask = test_prob.reshape(-1, height[layer_idx], width[layer_idx])
            # upsample
            test_map.append(
                F.interpolate(
                    test_mask.unsqueeze(1), size=self.image_size, mode="bilinear", align_corners=True
                ).squeeze()
            )
        # score aggregation
        score_map = torch.zeros_like(test_map[0])
        for layer_idx in range(len(self.pool_layers)):
            score_map += test_map[layer_idx]
        score_mask = score_map
        # invert probs to anomaly scores
        anomaly_map = score_mask.max() - score_mask

        return anomaly_map

    def __call__(self, **kwargs: Union[List[Tensor], List[int], List[List]]) -> Tensor:
        """Returns anomaly_map.

        Expects `distribution`, `height` and 'width' keywords to be passed explicitly

        Example
        >>> anomaly_map_generator = AnomalyMapGenerator(image_size=tuple(hparams.model.input_size),
        >>>        pool_layers=pool_layers)
        >>> output = self.anomaly_map_generator(distribution=dist, height=height, width=width)

        Raises:
            ValueError: `distribution`, `height` and 'width' keys are not found

        Returns:
            torch.Tensor: anomaly map
        """
        if not ("distribution" in kwargs and "height" in kwargs and "width" in kwargs):
            raise KeyError(f"Expected keys `distribution`, `height` and `width`. Found {kwargs.keys()}")

        # placate mypy
        distribution: List[Tensor] = cast(List[Tensor], kwargs["distribution"])
        height: List[int] = cast(List[int], kwargs["height"])
        width: List[int] = cast(List[int], kwargs["width"])
        return self.compute_anomaly_map(distribution, height, width)
