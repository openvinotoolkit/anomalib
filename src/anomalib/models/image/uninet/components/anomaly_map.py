"""Utilities for computing anomaly maps."""

# Original Code
# Copyright (c) 2025 Shun Wei
# https://github.com/pangdatangtt/UniNet
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.nn import functional as F  # noqa: N812

from anomalib.models.components.filters import GaussianBlur2d


def weighted_decision_mechanism(
    batch_size: int,
    output_list: list[torch.Tensor],
    alpha: float,
    beta: float,
    output_size: tuple[int, int] = (256, 256),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute anomaly maps using weighted decision mechanism.

    Args:
        batch_size (int): Batch size.
        output_list (list[torch.Tensor]): List of output tensors, each with shape [batch_size, H, W].
        alpha (float): Alpha parameter. Used for controlling the upper limit
        beta (float): Beta parameter. Used for controlling the lower limit
        output_size (tuple[int, int]): Output size.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Anomaly score and anomaly map.
    """
    # The original per-image top-k (k derived from alpha/beta weights) only ever read the
    # first, i.e. largest, value, so the score is the max of the blurred map. Computing it
    # directly is equivalent and avoids data-dependent shapes that dynamo cannot export.
    del alpha, beta
    device = output_list[0].device
    gaussian_blur = GaussianBlur2d(sigma=4.0, kernel_size=(5, 5), channels=1).to(device)

    # Process anomaly maps using tensor operations
    # Pre-allocate the processed anomaly maps tensor
    processed_anomaly_maps = torch.zeros(batch_size, *output_size, device=device)

    # Process each output tensor separately due to different spatial dimensions
    for output_tensor in output_list:
        # Interpolate current output to target size
        # Add channel dimension for interpolation: [batch_size, H, W] -> [batch_size, 1, H, W]
        output_resized = F.interpolate(
            output_tensor.unsqueeze(1),
            output_size,
            mode="bilinear",
            align_corners=True,
        ).squeeze(1)  # [batch_size, H_out, W_out]

        # Add to accumulated anomaly maps
        processed_anomaly_maps += output_resized

    anomaly_scores = gaussian_blur(processed_anomaly_maps.unsqueeze(1)).flatten(1).amax(dim=1)
    return anomaly_scores.detach().unsqueeze(1), processed_anomaly_maps.detach()
