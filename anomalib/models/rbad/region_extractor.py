"""Region-based Anomaly Detection with Real Time Training and Analysis.

Region Extractor.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.detection as detection
from torch import Tensor
from torchvision.ops import boxes as box_ops

from anomalib.models.rbad.region import (
    convert_to_patch_boxes,
    update_box_sizes_following_image_resize,
)


class RegionExtractor(nn.Module):
    def __init__(self, stage="rcnn", patch_mode=False, patch_size=32, use_original=False, **kwargs):
        # kwargs gives the configurable parameters
        # kwargs.keys() == {'max_overlap', 'min_size', 'likelihood'}

        assert stage in ["rpn", "rcnn"]

        super(RegionExtractor, self).__init__()

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pseudo_scores = torch.arange(1000, 0, -1, dtype=torch.float32, device=device)

        # Affects global behaviour of the extractor
        self.use_original = use_original
        self.stage = stage
        self.patch_mode = patch_mode

        self.min_size = 25 if "min_size" not in kwargs else kwargs["min_size"]
        self.stage_nms_thresh = 0.3 if "max_overlap" not in kwargs else kwargs["max_overlap"]
        self.box_confidence = 0.2 if "likelihood" not in kwargs else likelihood_to_class_threshold(kwargs["likelihood"])

        # Affects operation only when stage='rcnn'
        self.rcnn_score_thresh = self.box_confidence
        self.rcnn_detections_per_img = 100

        # Affects operation only when patch_mode == True
        self.patch_side_length = patch_size
        self.patch_nms_thresh = 0.3

        # Model and model components
        self.base_model = detection.fasterrcnn_resnet50_fpn(
            pretrained=True, rpn_pre_nms_top_n_test=1000, rpn_nms_thresh=0.7, rpn_post_nms_top_n_test=1000
        )

        self.transform = self.base_model.transform
        self.backbone = self.base_model.backbone
        self.rpn = self.base_model.rpn

        self.box_roi_pool = self.base_model.roi_heads.box_roi_pool
        self.box_head = self.base_model.roi_heads.box_head
        self.box_predictor = self.base_model.roi_heads.box_predictor

        self.box_coder = self.base_model.roi_heads.box_coder

    @torch.no_grad()
    def forward(self, input_tensor: Union[Tensor, List[Tensor]]):

        if self.training:
            raise ValueError("Should not be in training mode")

        output_boxes: List[Tensor] = []

        if self.use_original:
            predictions = self.base_model(input_tensor)
            output_boxes = [prediction["boxes"] for prediction in predictions]
            return output_boxes
        else:
            original_image_sizes = [image.shape[-2:] for image in input_tensor]
            images, targets = self.transform(input_tensor)
            transformed_image_sizes = images.image_sizes

            features = self.backbone(images.tensors)
            proposals = self.rpn(images, features, targets)[0]

            if self.stage == "rpn":
                for boxes, original_image_size, new_image_size in zip(
                    proposals, original_image_sizes, transformed_image_sizes
                ):
                    boxes = box_ops.clip_boxes_to_image(boxes, new_image_size)

                    keep = box_ops.remove_small_boxes(boxes, min_size=self.min_size)
                    boxes = boxes[keep]

                    keep = box_ops.nms(boxes, self.pseudo_scores[: boxes.shape[0]], self.stage_nms_thresh)
                    boxes = boxes[keep]

                    boxes = update_box_sizes_following_image_resize(boxes, new_image_size, original_image_size)

                    if self.patch_mode:
                        boxes = convert_to_patch_boxes(boxes, original_image_size, self.patch_side_length)

                        keep = box_ops.nms(boxes, self.pseudo_scores[: boxes.shape[0]], self.patch_nms_thresh)
                        boxes = boxes[keep]

                    output_boxes.append(boxes)
            elif self.stage == "rcnn":
                box_features = self.box_roi_pool(features, proposals, transformed_image_sizes)
                box_features = self.box_head(box_features)
                class_logits, box_regression = self.box_predictor(box_features)
                boxes_list, _, _ = self.postprocess_detections(
                    class_logits, box_regression, proposals, transformed_image_sizes
                )

                for boxes, original_image_size, new_image_size in zip(
                    boxes_list, original_image_sizes, transformed_image_sizes
                ):
                    boxes = update_box_sizes_following_image_resize(boxes, new_image_size, original_image_size)

                    if self.patch_mode:
                        boxes = convert_to_patch_boxes(boxes, original_image_size, self.patch_side_length)

                        keep = box_ops.nms(boxes, self.pseudo_scores[: boxes.shape[0]], self.patch_nms_thresh)
                        boxes = boxes[keep]

                    output_boxes.append(boxes)
            else:
                raise ValueError("Unknown stage {}".format(self.stage))

            return output_boxes

    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []

        for boxes, scores, image_shape in zip(pred_boxes, pred_scores, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.rcnn_score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove small boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=self.min_size)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, all boxes together
            keep = box_ops.nms(boxes, scores, self.stage_nms_thresh)

            # keep only topk scoring predictions
            keep = keep[: self.rcnn_detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels
