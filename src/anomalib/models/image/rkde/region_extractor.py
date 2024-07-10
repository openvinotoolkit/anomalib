"""Region-based Anomaly Detection with Real Time Training and Analysis.

Region Extractor.
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

import torch
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import boxes as box_ops

from anomalib.data.utils.boxes import scale_boxes


class RoiStage(str, Enum):
    """Processing stage from which rois are extracted."""

    RCNN = "rcnn"
    RPN = "rpn"


class RegionExtractor(nn.Module):
    """Extracts regions from the image.

    Args:
        stage (RoiStage, optional): Processing stage from which rois are extracted.
            Defaults to ``RoiStage.RCNN``.
        score_threshold (float, optional): Mimumum confidence score for the region proposals.
            Defaults to ``0.001``.
        min_size (int, optional): Minimum size in pixels for the region proposals.
            Defaults to ``25``.
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

        # Affects global behaviour of the region extractor
        self.stage = stage
        self.min_size = min_size
        self.iou_threshold = iou_threshold
        self.max_detections_per_image = max_detections_per_image

        # Affects behaviour depending on roi stage
        rpn_top_n = max_detections_per_image if self.stage == RoiStage.RPN else 1000
        rpn_score_thresh = score_threshold if self.stage == RoiStage.RPN else 0.0

        # Create the model
        self.faster_rcnn = fasterrcnn_resnet50_fpn(
            pretrained=True,
            rpn_post_nms_top_n_test=rpn_top_n,
            rpn_score_thresh=rpn_score_thresh,
            box_score_thresh=score_threshold,
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
            Tensor: Predicted regions, tensor of shape [N, 5] where N is the number of predicted regions in the batch,
                 and where each row describes the index of the image in the batch and the 4 bounding box coordinates.
        """
        if self.training:
            msg = "Should not be in training mode"
            raise ValueError(msg)

        if self.stage == RoiStage.RCNN:
            # get rois from rcnn output
            predictions = self.faster_rcnn(batch)
            all_regions = [prediction["boxes"] for prediction in predictions]
            all_scores = [prediction["scores"] for prediction in predictions]
        elif self.stage == RoiStage.RPN:
            # get rois from region proposal network
            images, _ = self.faster_rcnn.transform(batch)
            features = self.faster_rcnn.backbone(images.tensors)
            proposals, _ = self.faster_rcnn.rpn(images, features)
            # post-process raw rpn predictions
            all_regions = [box_ops.clip_boxes_to_image(boxes, images.tensors.shape[-2:]) for boxes in proposals]
            all_regions = [scale_boxes(boxes, images.tensors.shape[-2:], batch.shape[-2:]) for boxes in all_regions]
            all_scores = [torch.ones(boxes.shape[0]).to(boxes.device) for boxes in all_regions]
        else:
            msg = f"Unknown region extractor stage: {self.stage}"
            raise ValueError(msg)

        regions = self.post_process_box_predictions(all_regions, all_scores)

        # convert from list of [N, 4] tensors to single [N, 5] tensor where each row is [index-in-batch, x1, y1, x2, y2]
        indices = torch.repeat_interleave(
            torch.arange(len(regions)),
            torch.Tensor([rois.shape[0] for rois in regions]).int(),
        )
        return torch.cat([indices.unsqueeze(1).to(batch.device), torch.cat(regions)], dim=1)

    def post_process_box_predictions(self, pred_boxes: torch.Tensor, pred_scores: torch.Tensor) -> list[torch.Tensor]:
        """Post-processes the box predictions.

        The post-processing consists of removing small boxes, applying nms, and
        keeping only the k boxes with the highest confidence score.

        Args:
            pred_boxes (torch.Tensor): Box predictions of shape (N, 4).
            pred_scores (torch.Tensor): torch.Tensor of shape () with a confidence score for each box prediction.

        Returns:
            list[torch.Tensor]: Post-processed box predictions of shape (N, 4).
        """
        processed_boxes_list: list[torch.Tensor] = []
        for boxes, scores in zip(pred_boxes, pred_scores, strict=True):
            # remove small boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=self.min_size)
            processed_boxes, processed_scores = boxes[keep], scores[keep]

            # non-maximum suppression, all boxes together
            keep = box_ops.nms(processed_boxes, processed_scores, self.iou_threshold)

            # keep only top-k scoring predictions
            keep = keep[: self.max_detections_per_image]
            processed_boxes = processed_boxes[keep]

            processed_boxes_list.append(processed_boxes)

        return processed_boxes_list
