"""Anomaly Map Generator for CS-Flow model."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.jit import script_if_tracing


class AnomalyMapMode(str, Enum):
    """Generate anomaly map from all the scales or the max."""

    ALL = "all"
    MAX = "max"


class AnomalyMapGenerator(nn.Module):
    """Anomaly Map Generator for CS-Flow model.

    Args:
        input_dims (tuple[int, int, int]): Input dimensions.
        mode (AnomalyMapMode): Anomaly map mode. Defaults to AnomalyMapMode.ALL.
    """

    def __init__(self, input_dims: tuple[int, int, int], mode: AnomalyMapMode = AnomalyMapMode.ALL) -> None:
        super().__init__()
        self.mode = mode
        self.input_dims = input_dims

    @staticmethod
    @script_if_tracing
    def generate_empty_anomaly_map(inputs: Tensor, h: int, w: int) -> Tensor:
        """Generate empty anomaly map.

        Args:
            batch_size (torch.Tensor): Batch size.
            input_dims (Tuple[int, int, int]): Input dimensions.

        Returns:
            Tensor: Empty anomaly map.
        """
        return torch.ones(inputs.shape[0], 1, h, w, device=inputs.device)

    def forward(self, inputs: Tensor) -> Tensor:
        """Get anomaly maps by taking mean of the z-distributions across channels.

        By default it computes anomaly maps for all the scales as it gave better performance on initial tests.
        Use ``AnomalyMapMode.MAX`` for the largest scale as mentioned in the paper.

        Args:
            inputs (Tensor): z-distributions for the three scales.
            mode (AnomalyMapMode): Anomaly map mode.

        Returns:
            Tensor: Anomaly maps.
        """
        anomaly_map: Tensor
        if self.mode == AnomalyMapMode.ALL:
            anomaly_map = self.generate_empty_anomaly_map(inputs[0], self.input_dims[1], self.input_dims[2])
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
