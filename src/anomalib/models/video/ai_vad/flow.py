"""Optical Flow extraction module for AI-VAD implementation.

This module implements the optical flow extraction stage of the AI-VAD model. It uses
RAFT (Recurrent All-Pairs Field Transforms) to compute dense optical flow between
consecutive video frames.

Example:
    >>> from anomalib.models.video.ai_vad.flow import FlowExtractor
    >>> import torch
    >>> extractor = FlowExtractor()
    >>> first_frame = torch.randn(32, 3, 256, 256)  # (N, C, H, W)
    >>> last_frame = torch.randn(32, 3, 256, 256)  # (N, C, H, W)
    >>> flow = extractor(first_frame, last_frame)
    >>> flow.shape
    torch.Size([32, 2, 256, 256])

The module provides the following components:
    - :class:`FlowExtractor`: Main class that handles optical flow computation using
      RAFT model
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import torchvision.transforms.functional as F  # noqa: N812
from torch import nn
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large


class FlowExtractor(nn.Module):
    """Optical Flow extractor.

    Computes the pixel displacement between 2 consecutive frames from a video clip.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=weights)
        self.transforms = weights.transforms()

    def pre_process(self, first_frame: torch.Tensor, last_frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Resize inputs to dimensions required by backbone.

        Args:
            first_frame (torch.Tensor): Starting frame of optical flow computation.
            last_frame (torch.Tensor): Last frame of optical flow computation.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Preprocessed first and last frame.
        """
        first_frame = F.resize(first_frame, size=[520, 960], antialias=False)
        last_frame = F.resize(last_frame, size=[520, 960], antialias=False)
        return self.transforms(first_frame, last_frame)

    def forward(self, first_frame: torch.Tensor, last_frame: torch.Tensor) -> torch.Tensor:
        """Forward pass through the flow extractor.

        Args:
            first_frame (torch.Tensor): Batch of starting frames of shape (N, 3, H, W).
            last_frame (torch.Tensor): Batch of last frames of shape (N, 3, H, W).

        Returns:
            Tensor: Estimated optical flow map of shape (N, 2, H, W).
        """
        height, width = first_frame.shape[-2:]

        # preprocess batch
        first_frame, last_frame = self.pre_process(first_frame, last_frame)

        # get flow maps
        with torch.no_grad():
            flows = self.model(first_frame, last_frame)[-1]

        # convert back to original size
        return F.resize(flows, [height, width], antialias=False)
