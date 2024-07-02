"""Anomaly Map Generator for CFlow model implementation."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap."""

    def __init__(
        self,
        pool_layers: Sequence[str],
    ) -> None:
        super().__init__()
        self.distance = torch.nn.PairwiseDistance(p=2, keepdim=True)
        self.pool_layers: Sequence[str] = pool_layers

    def compute_anomaly_map(
        self,
        distribution: list[torch.Tensor],
        height: list[int],
        width: list[int],
        image_size: tuple[int, int] | torch.Size | None,
    ) -> torch.Tensor:
        """Compute the layer map based on likelihood estimation.

        Args:
            distribution (list[torch.Tensor]): List of likelihoods for each layer.
            height (list[int]): List of heights of the feature maps.
            width (list[int]): List of widths of the feature maps.
            image_size (tuple[int, int] | torch.Size | None): Size of the input image.

        Returns:
          Final Anomaly Map

        """
        layer_maps: list[torch.Tensor] = []
        for layer_idx in range(len(self.pool_layers)):
            layer_distribution = distribution[layer_idx].clone().detach()
            # Normalize the likelihoods to (-Inf:0] and convert to probs in range [0:1]
            layer_probabilities = torch.exp(layer_distribution - layer_distribution.max())
            layer_map = layer_probabilities.reshape(-1, height[layer_idx], width[layer_idx])
            # upsample
            if image_size is not None:
                layer_map = F.interpolate(
                    layer_map.unsqueeze(1),
                    size=image_size,
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(1)
            layer_maps.append(layer_map)
        # score aggregation
        score_map = torch.zeros_like(layer_maps[0])
        for layer_idx in range(len(self.pool_layers)):
            score_map += layer_maps[layer_idx]

        # Invert probs to anomaly scores
        return score_map.max() - score_map

    def forward(self, **kwargs: list[torch.Tensor] | list[int] | list[list]) -> torch.Tensor:
        """Return anomaly_map.

        Expects `distribution`, `height` and 'width' keywords to be passed explicitly

        Example:
            >>> anomaly_map_generator = AnomalyMapGenerator(image_size=tuple(hparams.model.input_size),
            >>>        pool_layers=pool_layers)
            >>> output = self.anomaly_map_generator(distribution=dist, height=height, width=width)

        Raises:
            ValueError: `distribution`, `height` and 'width' keys are not found

        Returns:
            torch.Tensor: anomaly map
        """
        if not ("distribution" in kwargs and "height" in kwargs and "width" in kwargs):
            msg = f"Expected keys `distribution`, `height` and `width`. Found {kwargs.keys()}"
            raise KeyError(msg)

        # placate mypy
        distribution: list[torch.Tensor] = cast(list[torch.Tensor], kwargs["distribution"])
        height: list[int] = cast(list[int], kwargs["height"])
        width: list[int] = cast(list[int], kwargs["width"])
        image_size: tuple[int, int] | torch.Size | None = kwargs.get("image_size", None)
        return self.compute_anomaly_map(distribution, height, width, image_size)
