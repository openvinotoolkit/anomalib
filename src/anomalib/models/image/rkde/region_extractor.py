"""Region-based Anomaly Detection with Real Time Training and Analysis.

Region Extractor.
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

import torch
from torch import nn
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.ops import box_area, nms


class RoiStage(str, Enum):
    """Processing stage from which rois are extracted."""

    RCNN = "rcnn"
    RPN = "rpn"


class RegionExtractor(nn.Module):
    """Extracts regions from the image.

    Args:
        stage (RoiStage, optional): Processing stage from which rois are extracted.
            Defaults to ``RoiStage.RCNN``.
        score_threshold (float, optional): Minimum confidence score for the region proposals.
            Defaults to ``0.001``.
        min_size (int, optional): Minimum size in pixels for the region proposals.
            Defaults to ``100``.
        iou_threshold (float, optional): Intersection-Over-Union threshold used during NMS.
            Defaults to ``0.3``.
        max_detections_per_image (int, optional): Maximum number of region proposals per image.
            Defaults to ``100``.
    """

    def __init__(
        self,
        stage: RoiStage = RoiStage.RCNN,
        score_threshold: float = 0.001,
        min_size: int = 25,
        iou_threshold: float = 0.3,
        max_detections_per_image: int = 100,
    ) -> None:
        super().__init__()
        # Affects global behaviou
        self.stage = stage
        self.min_size = min_size
        self.iou_threshold = iou_threshold
        self.max_detections_per_image = max_detections_per_image

        # Affects behaviour depending on roi stage
        rpn_top_n = max_detections_per_image if self.stage == RoiStage.RPN else 1000
        rpn_score_thresh = score_threshold if self.stage == RoiStage.RPN else 0.0

        self.backbone = maskrcnn_resnet50_fpn_v2(
            weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
            rpn_post_nms_top_n_test=rpn_top_n,
            rpn_score_thresh=rpn_score_thresh,
            box_score_thresh=score_threshold,
            rpn_nms_thresh=0.3,
            box_nms_thresh=1.0,  # this disables nms (we apply custom label-agnostic nms during post-processing)
            box_detections_per_img=1000,  # this disables filtering top-k predictions (we apply our own after nms)
        )

    @torch.no_grad()
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            batch (torch.Tensor): Batch of input images of shape [B, C, H, W].

        Raises:
            ValueError: When ``stage`` is not one of ``rcnn`` or ``rpn``.

        Returns:
            List of dictionaries containing the processed box predictions.
            The dictionary has the keys ``boxes``, ``masks``, ``scores``, and ``labels``.
        """
        if self.training:
            msg = "Should not be in training mode"
            raise ValueError(msg)

        regions: list[dict[str, torch.Tensor]] = self.backbone(batch)

        return self.post_process_box_predictions(regions)

    def post_process_box_predictions(self, regions: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
        """Post-processes the box predictions.

        The post-processing consists of removing small boxes, applying nms, and
        keeping only the k boxes with the highest confidence score.

        Args:
            regions (list[dict[str, torch.Tensor]]): List of dictionaries containing the box predictions.

        Returns:
            list[dict[str, torch.Tensor]]: List of dictionaries containing the processed box predictions.
        """
        new_regions = []
        for _region in regions:
            boxes = _region["boxes"]
            masks = _region["masks"]
            scores = _region["scores"]
            labels = _region["labels"]
            # remove small boxes
            keep = torch.where(box_area(boxes) > self.min_size)
            boxes = boxes[keep]
            masks = masks[keep]
            scores = scores[keep]
            labels = labels[keep]
            # # non-maximum suppression
            keep = nms(boxes, scores, self.iou_threshold)
            # # keep only top-k scoring predictions
            keep = keep[: self.max_detections_per_image]
            processed_boxes = {
                "boxes": boxes[keep],
                "masks": masks[keep],
                "scores": scores[keep],
                "labels": labels[keep],
            }
            new_regions.append(processed_boxes)

        return new_regions
