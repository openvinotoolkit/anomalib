"""Anomaly Map Generator for CFlow model implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple, Union, cast

import torch
import torch.nn.functional as F
from omegaconf import ListConfig
from torch import Tensor, nn


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap."""

    def __init__(
        self,
        image_size: Union[ListConfig, Tuple],
        pool_layers: List[str],
    ):
        super().__init__()
        self.distance = torch.nn.PairwiseDistance(p=2, keepdim=True)
        self.image_size = image_size if isinstance(image_size, tuple) else tuple(image_size)
        self.pool_layers: List[str] = pool_layers

    def compute_anomaly_map(self, distribution: List[Tensor], height: List[int], width: List[int]) -> Tensor:
        """Compute the layer map based on likelihood estimation.

        Args:
          distribution: Probability distribution for each decoder block
          height: blocks height
          width: blocks width

        Returns:
          Final Anomaly Map

        """
        layer_maps: List[Tensor] = []
        for layer_idx in range(len(self.pool_layers)):
            layer_distribution = distribution[layer_idx].clone().detach()
            # Normalize the likelihoods to (-Inf:0] and convert to probs in range [0:1]
            layer_probabilities = torch.exp(layer_distribution - layer_distribution.max())
            layer_map = layer_probabilities.reshape(-1, height[layer_idx], width[layer_idx])
            # upsample
            layer_maps.append(
                F.interpolate(
                    layer_map.unsqueeze(1), size=self.image_size, mode="bilinear", align_corners=True
                ).squeeze(1)
            )
        # score aggregation
        score_map = torch.zeros_like(layer_maps[0])
        for layer_idx in range(len(self.pool_layers)):
            score_map += layer_maps[layer_idx]

        # Invert probs to anomaly scores
        anomaly_map = score_map.max() - score_map

        return anomaly_map

    def forward(self, **kwargs: Union[List[Tensor], List[int], List[List]]) -> Tensor:
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
