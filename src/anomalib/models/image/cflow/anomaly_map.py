"""Anomaly Map Generator for CFlow model implementation.

This module provides the anomaly map generation functionality for the CFlow model.
The generator takes feature distributions from multiple layers and combines them
into a single anomaly heatmap.

Example:
    >>> from anomalib.models.image.cflow.anomaly_map import AnomalyMapGenerator
    >>> import torch
    >>> # Initialize generator
    >>> pool_layers = ["layer1", "layer2", "layer3"]
    >>> generator = AnomalyMapGenerator(pool_layers=pool_layers)
    >>> # Generate anomaly map
    >>> distribution = [torch.randn(32, 64) for _ in range(3)]
    >>> height = [32, 16, 8]
    >>> width = [32, 16, 8]
    >>> anomaly_map = generator(
    ...     distribution=distribution,
    ...     height=height,
    ...     width=width
    ... )
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


class AnomalyMapGenerator(nn.Module):
    """Generate anomaly heatmap from layer-wise feature distributions.

    The generator combines likelihood estimations from multiple feature layers into
    a single anomaly heatmap by upsampling and aggregating the scores.

    Args:
        pool_layers (Sequence[str]): Names of pooling layers from which to extract
            features.

    Example:
        >>> pool_layers = ["layer1", "layer2", "layer3"]
        >>> generator = AnomalyMapGenerator(pool_layers=pool_layers)
        >>> distribution = [torch.randn(32, 64) for _ in range(3)]
        >>> height = [32, 16, 8]
        >>> width = [32, 16, 8]
        >>> anomaly_map = generator(
        ...     distribution=distribution,
        ...     height=height,
        ...     width=width
        ... )
    """

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
        """Compute anomaly map from layer-wise likelihood distributions.

        The method normalizes likelihood scores from each layer, upsamples them to
        a common size, and combines them into a final anomaly map.

        Args:
            distribution (list[torch.Tensor]): List of likelihood distributions for
                each layer.
            height (list[int]): List of feature map heights for each layer.
            width (list[int]): List of feature map widths for each layer.
            image_size (tuple[int, int] | torch.Size | None): Target size for the
                output anomaly map. If None, keeps the original size.

        Returns:
            torch.Tensor: Anomaly map tensor where higher values indicate higher
                likelihood of anomaly.
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
        """Generate anomaly map from input feature distributions.

        The method expects keyword arguments containing the feature distributions
        and corresponding spatial dimensions.

        Args:
            **kwargs: Keyword arguments containing:
                - distribution (list[torch.Tensor]): Feature distributions
                - height (list[int]): Feature map heights
                - width (list[int]): Feature map widths
                - image_size (tuple[int, int] | torch.Size | None, optional):
                    Target output size

        Example:
            >>> generator = AnomalyMapGenerator(pool_layers=["layer1", "layer2"])
            >>> distribution = [torch.randn(32, 64) for _ in range(2)]
            >>> height = [32, 16]
            >>> width = [32, 16]
            >>> anomaly_map = generator(
            ...     distribution=distribution,
            ...     height=height,
            ...     width=width
            ... )

        Raises:
            KeyError: If required arguments `distribution`, `height` or `width`
                are missing.

        Returns:
            torch.Tensor: Generated anomaly map.
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
