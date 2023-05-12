"""Regions extraction module of AI-VAD model implementation."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2


class RegionExtractor(nn.Module):
    """Region extractor for AI-VAD.

    Args:
        box_score_thresh (float): Confidence threshold for bounding box predictions.
    """

    def __init__(self, box_score_thresh: float = 0.8) -> None:
        super().__init__()

        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.backbone = maskrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=box_score_thresh, rpn_nms_thresh=0.3)

    def forward(self, batch: Tensor) -> list[dict]:
        """Forward pass through region extractor.

        Args:
            batch (Tensor): Batch of input images of shape (N, C, H, W)
        Returns:
            list[dict]: List of Mask RCNN predictions for each image in the batch.
        """
        with torch.no_grad():
            regions = self.backbone(batch)

        return regions
