"""FastFlow Anomaly Map Generator Implementation.

This module implements the anomaly map generation for the FastFlow model. The
generator takes hidden variables from normalizing flow blocks and produces an
anomaly heatmap by computing flow maps.

Example:
    >>> from anomalib.models.image.fastflow.anomaly_map import AnomalyMapGenerator
    >>> generator = AnomalyMapGenerator(input_size=(256, 256))
    >>> hidden_vars = [torch.randn(1, 64, 32, 32)]  # from NF blocks
    >>> anomaly_map = generator(hidden_vars)  # returns anomaly heatmap
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from omegaconf import ListConfig
from torch import nn
from torch.nn import functional as F  # noqa: N812


class AnomalyMapGenerator(nn.Module):
    """Generate anomaly heatmaps from FastFlow hidden variables.

    The generator takes hidden variables from normalizing flow blocks and produces
    an anomaly heatmap. For each hidden variable tensor, it:
        1. Computes negative log probability
        2. Converts to probability via exponential
        3. Interpolates to input size
        4. Stacks and averages flow maps to produce final anomaly map

    Args:
        input_size (ListConfig | tuple): Target size for the anomaly map as
            ``(height, width)``. If ``ListConfig`` is provided, it will be
            converted to tuple.

    Example:
        >>> generator = AnomalyMapGenerator(input_size=(256, 256))
        >>> hidden_vars = [torch.randn(1, 64, 32, 32)]  # from NF blocks
        >>> anomaly_map = generator(hidden_vars)
        >>> anomaly_map.shape
        torch.Size([1, 1, 256, 256])
    """

    def __init__(self, input_size: ListConfig | tuple) -> None:
        super().__init__()
        self.input_size = input_size if isinstance(input_size, tuple) else tuple(input_size)

    def forward(self, hidden_variables: list[torch.Tensor]) -> torch.Tensor:
        """Generate anomaly heatmap from hidden variables.

        This implementation generates the heatmap based on the flow maps computed
        from the normalizing flow (NF) FastFlow blocks. Each block yields a flow
        map, which overall is stacked and averaged to produce an anomaly map.

        The process for each hidden variable is:
            1. Compute negative log probability as mean of squared values
            2. Convert to probability via exponential
            3. Interpolate to input size
            4. Stack all flow maps and average to get final anomaly map

        Args:
            hidden_variables (list[torch.Tensor]): List of hidden variables from
                each NF FastFlow block. Each tensor has shape
                ``(N, C, H, W)``.

        Returns:
            torch.Tensor: Anomaly heatmap with shape ``(N, 1, H, W)`` where
                ``H, W`` match the ``input_size``.
        """
        flow_maps: list[torch.Tensor] = []
        for hidden_variable in hidden_variables:
            log_prob = -torch.mean(hidden_variable**2, dim=1, keepdim=True) * 0.5
            prob = torch.exp(log_prob)
            flow_map = F.interpolate(
                input=-prob,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )
            flow_maps.append(flow_map)
        flow_maps = torch.stack(flow_maps, dim=-1)
        return torch.mean(flow_maps, dim=-1)
