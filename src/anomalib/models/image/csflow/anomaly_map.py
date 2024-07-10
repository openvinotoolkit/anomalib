"""Anomaly Map Generator for CS-Flow model."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


class AnomalyMapMode(str, Enum):
    """Generate anomaly map from all the scales or the max."""

    ALL = "all"
    MAX = "max"


class AnomalyMapGenerator(nn.Module):
    """Anomaly Map Generator for CS-Flow model.

    Args:
        input_dims (tuple[int, int, int]): Input dimensions.
        mode (AnomalyMapMode): Anomaly map mode.
            Defaults to ``AnomalyMapMode.ALL``.
    """

    def __init__(self, input_dims: tuple[int, int, int], mode: AnomalyMapMode = AnomalyMapMode.ALL) -> None:
        super().__init__()
        self.mode = mode
        self.input_dims = input_dims

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Get anomaly maps by taking mean of the z-distributions across channels.

        By default it computes anomaly maps for all the scales as it gave better performance on initial tests.
        Use ``AnomalyMapMode.MAX`` for the largest scale as mentioned in the paper.

        Args:
            inputs (torch.Tensor): z-distributions for the three scales.
            mode (AnomalyMapMode): Anomaly map mode.

        Returns:
            Tensor: Anomaly maps.
        """
        anomaly_map: torch.Tensor
        if self.mode == AnomalyMapMode.ALL:
            anomaly_map = torch.ones(inputs[0].shape[0], 1, *self.input_dims[1:]).to(inputs[0].device)
            for z_dist in inputs:
                mean_z = (z_dist**2).mean(dim=1, keepdim=True)
                anomaly_map *= F.interpolate(
                    mean_z,
                    size=self.input_dims[1:],
                    mode="bilinear",
                    align_corners=False,
                )
        else:
            mean_z = (inputs[0] ** 2).mean(dim=1, keepdim=True)
            anomaly_map = F.interpolate(
                mean_z,
                size=self.input_dims[1:],
                mode="bilinear",
                align_corners=False,
            )

        return anomaly_map
