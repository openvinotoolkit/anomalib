"""Regions extraction module of AI-VAD model implementation."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.ops import box_area

PERSON_LABEL = 1


class RegionExtractor(nn.Module):
    """Region extractor for AI-VAD.

    Args:
        box_score_thresh (float): Confidence threshold for bounding box predictions.
    """

    def __init__(self, box_score_thresh: float = 0.8) -> None:
        super().__init__()

        self.persons_only = False
        self.min_bbox_area = 100
        self.max_overlap = 0.65

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

        regions = self.post_process_bbox_detections(regions)

        return regions

    def post_process_bbox_detections(self, batch_regions):
        filtered_regions = []
        for im_regions in batch_regions:
            if self.only_persons:
                im_regions = self._keep_only_persons(im_regions)
            im_regions = self._filter_by_area(im_regions, self.min_bbox_area)
            im_regions = self._delete_overlapping_boxes(im_regions, self.max_overlap)
            filtered_regions.append(im_regions)

    def _keep_only_persons(self, regions):
        keep = torch.where(regions["labels"] == PERSON_LABEL)
        return self.subsample_regions(regions, keep)

    def _filter_by_area(self, regions, min_area):
        """Remove all regions with a surface area smaller than the specified value."""

        areas = box_area(regions["boxes"])
        keep = torch.where(areas > min_area)
        return self.subsample_regions(regions, keep)

    def _delete_overlapping_boxes(regions, threshold):
        """Delete overlapping bounding boxes, larger boxes are kept."""

        areas = box_area(regions["boxes"])
        # sort by area
        areas.argsort()

    @staticmethod
    def subsample_regions(regions, indices):
        new_regions_dict = {}
        for key, value in regions.items():
            new_regions_dict[key] = value[indices]
        return new_regions_dict
