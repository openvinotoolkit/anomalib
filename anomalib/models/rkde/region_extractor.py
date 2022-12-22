"""Region-based Anomaly Detection with Real Time Training and Analysis.

Region Extractor.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, List, Optional

import torch
from torch import Tensor, nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import boxes as box_ops


class RegionExtractor(nn.Module):
    """Extracts regions from the image."""

    def __init__(
        self,
        stage: str = "rcnn",
        use_original: bool = True,
        min_size: int = 25,
        iou_threshold: float = 0.3,
        likelihood: Optional[float] = None,
        rcnn_detections_per_image: int = 100,
    ) -> None:
        super().__init__()

        # Affects global behaviour of the extractor
        self.stage = stage
        self.use_original = use_original
        self.min_size = min_size
        self.iou_threshold = iou_threshold

        # Affects operation only when stage='rcnn'
        self.rcnn_score_thresh = 0.2 if likelihood is None else self.likelihood_to_class_threshold(likelihood)
        self.max_detections_per_image = rcnn_detections_per_image

        # Model and model components
        self.faster_rcnn = fasterrcnn_resnet50_fpn(
            pretrained=True,
            box_score_thresh=self.rcnn_score_thresh,
            box_nms_thresh=1.0,  # this disables nms (we apply our own label-agnostic nms during post-processing)
            box_detections_per_img=1000,  # this disables filtering top k predictions (again, we apply our own version)
        )

        if self.stage == "rpn":
            self.transform_shape = Tensor([])
            self.proposals = Tensor([])
            self.faster_rcnn.transform.register_forward_hook(self.get_transform_shape_hook())
            self.faster_rcnn.rpn.register_forward_hook(self.get_proposals_hook())
            self.max_detections_per_image = 1000  # disable score-based filtering when in rpn mode

    def get_proposals_hook(self) -> Callable:
        """Forward hook that retrieves the outputs of the Region Proposal Network."""

        def hook(_, __, output):
            self.proposals = output[0]

        return hook

    def get_transform_shape_hook(self) -> Callable:
        """Forward hook that retrieves the size of the input images after the RCNN transform is applied."""

        def hook(_, __, output):
            self.transform_shape = output[0].tensors.shape[-2:]

        return hook

    @torch.no_grad()
    def forward(self, batch: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            input (Union[Tensor, List[Tensor]]): Input tensor or list of tensors.

        Raises:
            ValueError: When ``stage`` is not one of ``rcnn`` or ``rpn``.

        Returns:
            List[Tensor]: Regions, comprising ``List`` of boxes for each image.
        """
        if self.training:
            raise ValueError("Should not be in training mode")

        regions: List[Tensor] = []

        # forward pass through faster rcnn
        predictions = self.faster_rcnn(batch)

        if self.stage == "rcnn":
            # get boxes from model predictions
            all_regions = [prediction["boxes"] for prediction in predictions]
            all_scores = [prediction["scores"] for prediction in predictions]
        elif self.stage == "rpn":
            # get boxes from region proposals
            all_regions = [box_ops.clip_boxes_to_image(boxes, self.transform_shape) for boxes in self.proposals]
            all_regions = [self.scale_boxes(boxes, self.transform_shape, batch.shape[-2:]) for boxes in all_regions]
            all_scores = [torch.ones(boxes.shape[0]).to(boxes.device) for boxes in all_regions]
        else:
            raise ValueError(f"Unknown region extractor stage: {self.stage}")

        regions = self.post_process_box_predictions(all_regions, all_scores)

        indices = torch.repeat_interleave(torch.arange(len(regions)), Tensor([rois.shape[0] for rois in regions]).int())
        regions = torch.cat([indices.unsqueeze(1).to(batch.device), torch.cat(regions)], dim=1)
        return regions

    def post_process_box_predictions(self, pred_boxes: Tensor, pred_scores: Tensor) -> List[Tensor]:
        """Post-processes the box predictions.

        The post-processing consists of removing small boxes, applying nms, and
        keeping only the k boxes with the highest confidence score.

        Args:
            pred_boxes (Tensor): Box predictions of shape (N, 4).
            pred_scores (Tensor): Tensor of shape () with a confidence score for each box prediction.

        Returns:
            List[Tensor]: Post-processed box predictions of shape (N, 4).
        """

        processed_boxes: List[Tensor] = []
        for boxes, scores in zip(pred_boxes, pred_scores):

            # remove small boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=self.min_size)
            boxes, scores = boxes[keep], scores[keep]

            # non-maximum suppression, all boxes together
            keep = box_ops.nms(boxes, scores, self.iou_threshold)

            # keep only top-k scoring predictions
            keep = keep[: self.max_detections_per_image]
            boxes = boxes[keep]

            processed_boxes.append(boxes)

        return processed_boxes

    @staticmethod
    def likelihood_to_class_threshold(likelihood: float) -> float:
        """Convert likelihood to class threshold.

        Args:
            likelihood (float): Input likelihood.

        Returns:
            float: Class threshold.
        """
        threshold = (torch.cos(torch.tensor(likelihood) * torch.pi / 2.0) ** 4.0).item()
        return threshold

    @staticmethod
    def scale_boxes(boxes: Tensor, image_size: torch.Size, new_size: torch.Size) -> Tensor:
        """Scale bbox coordinates to a new image size.

        Args:
            boxes (Tensor): Boxes of shape (N, 4) - (x1, y1, x2, y2).
            image_size (Size): Size of the original image in which the bbox coordinates were retrieved.
            new_size (Size): New image size to which the bbox coordinates will be scaled.

        Returns:
            Tensor: Updated boxes of shape (N, 4) - (x1, y1, x2, y2).
        """
        scale = Tensor([*new_size]) / Tensor([*image_size])
        return boxes * scale.repeat(2).to(boxes.device)
