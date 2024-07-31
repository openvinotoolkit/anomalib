"""FastFlow Anomaly Map Generator Implementation."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from omegaconf import ListConfig
from torch import nn
from torch.nn import functional as F  # noqa: N812


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap.

    Args:
        input_size (ListConfig | tuple): Input size.
    """

    def __init__(self, input_size: ListConfig | tuple) -> None:
        super().__init__()
        self.input_size = input_size if isinstance(input_size, tuple) else tuple(input_size)

    def forward(self, hidden_variables: list[torch.Tensor]) -> torch.Tensor:
        """Generate Anomaly Heatmap.

        This implementation generates the heatmap based on the flow maps
        computed from the normalizing flow (NF) FastFlow blocks. Each block
        yields a flow map, which overall is stacked and averaged to an anomaly
        map.

        Args:
            hidden_variables (list[torch.Tensor]): List of hidden variables from each NF FastFlow block.

        Returns:
            Tensor: Anomaly Map.
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
