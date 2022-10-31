"""Region-based Anomaly Detection with Real Time Training and Analysis.

Region Extractor.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.models.detection as detection
from torch import Size, Tensor
from torch.utils.model_zoo import load_url
from torchvision.ops import RoIAlign
from torchvision.ops import boxes as box_ops


class FeatureExtractor(nn.Module):
    """Region-based Feature Extractor."""

    def __init__(
        self,
        region_extractor_stage: str = "rcnn",
        # TODO: Rename ``use_original``
        use_original: bool = False,
        min_box_size: int = 25,
        iou_threshold: float = 0.3,
        likelihood: Optional[float] = None,
        tiling: bool = False,
        tile_size: int = 32,
    ) -> None:
        super().__init__()

        self.__model = models.alexnet(pretrained=False)

        # TODO: Load this via torch url.
        state_dict = torch.load("rcnn_feature_extractor.pth", map_location="cpu")

        # Create the backbone.
        self.backbone = self.__model.features[:-1]
        self.backbone.load_state_dict(state_dict=state_dict["backbone"])

        # Create the region proposal network.
        self.region_extractor = RegionExtractor(
            stage=region_extractor_stage,
            use_original=use_original,
            min_size=min_box_size,
            iou_threshold=iou_threshold,
            likelihood=likelihood,
            tiling=tiling,
            tile_size=tile_size,
        ).eval()

        # Create RoI Align Network.
        self.roi_align = RoIAlign(output_size=(6, 6), spatial_scale=1 / 16, sampling_ratio=0)

        # Classifier network to extract the features.
        self.classifer = self.__model.classifier[:-1]
        self.classifer.load_state_dict(state_dict=state_dict["classifier"])

    @torch.no_grad()
    def forward(self, input: Tensor) -> Tensor:
        """Forward-Pass Method.

        Args:
            input (Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            Tensor: Region-based features extracted from the input tensor.
        """

        # Get the extracted regions from the region extractor.
        boxes = self.region_extractor(input)

        # Apply the feature extractor transforms
        input, scale = self.transform(input)

        # Process RoIs.
        boxes = [box.unsqueeze(0) for box in boxes]
        rois = torch.cat(boxes, dim=0)
        # Add zero column for the scores.
        rois = F.pad(input=rois, pad=(1, 0, 0, 0), mode="constant", value=0)
        # Scale the RoIs based on the the new image size.
        rois *= scale

        # Forward-pass through the backbone, RoI Align and classifier.
        features = self.backbone(input)  # n_rois x 256 x 6 x 6 (AlexNet)
        features = self.roi_align(features, rois.view(-1, 5))  # n_rois x 4096
        features = self.classifer(features.view(features.size(0), -1))

        return features

    def transform(self, input: Tensor) -> Tuple[Tensor, float]:
        """Apply the feature extractor transforms

        Args:
            input (Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            Tuple[Tensor, float]: Output tensor of shape (N, C, H, W) and the scale.
        """
        height, width = input.shape[2:]
        shorter_image_size, longer_image_size = min(height, width), max(height, width)
        target_image_size, max_image_size = 600, 1000

        scale = target_image_size / shorter_image_size
        if round(scale * longer_image_size) > max_image_size:
            print("WARNING: cfg.MAX_SIZE exceeded. Using a different scaling ratio")
            scale = max_image_size / longer_image_size

        resized_tensor = F.interpolate(input, scale_factor=scale, mode="bilinear", align_corners=False)

        # Apply the same transformation as the original model.
        mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(input.device)
        std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(input.device)
        normalized_tensor = (resized_tensor - mean) / std

        return normalized_tensor, scale


class RegionExtractor(nn.Module):
    """Extracts regions from the image."""

    def __init__(
        self,
        stage: str = "rcnn",
        use_original: bool = False,
        min_size: int = 25,
        iou_threshold: float = 0.3,
        likelihood: Optional[float] = None,
        tiling: bool = False,
        tile_size: int = 32,
    ) -> None:
        super(RegionExtractor, self).__init__()

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pseudo_scores = torch.arange(1000, 0, -1, dtype=torch.float32, device=device)

        # Affects global behaviour of the extractor
        self.stage = stage
        self.use_original = use_original
        self.min_size = min_size
        self.iou_threshold = iou_threshold

        # Affects operation only when stage='rcnn'
        self.rcnn_score_thresh = 0.2 if likelihood is None else likelihood_to_class_threshold(likelihood)
        self.rcnn_detections_per_img = 100

        # Affects operation only when patch_mode == True
        self.tiling = tiling
        self.tile_size = tile_size
        self.tile_iou_threshold = 0.3

        # Model and model components
        self.faster_rcnn = detection.fasterrcnn_resnet50_fpn(
            pretrained=True, rpn_pre_nms_top_n_test=1000, rpn_nms_thresh=0.7, rpn_post_nms_top_n_test=1000
        )

    @torch.no_grad()
    def forward(self, input: Union[Tensor, List[Tensor]]) -> List[Tensor]:
        """Forward pass of the model.

        Args:
            input (Union[Tensor, List[Tensor]]): Input tensor or list of tensors.

        Raises:
            ValueError: When the model is not in the correct mode.
            ValueError: When ``stage`` is not one of ``rcnn`` or ``rpn``.

        Returns:
            List[Tensor]: Regions, comprising ``List`` of boxes for each image.
        """
        if self.training:
            raise ValueError("Should not be in training mode")

        regions: List[Tensor] = []

        if self.use_original:
            predictions = self.faster_rcnn(input)
            regions = [prediction["boxes"] for prediction in predictions]
            return regions
        else:
            original_image_sizes = [image.shape[-2:] for image in input]
            images, targets = self.faster_rcnn.transform(input)
            transformed_image_sizes = images.image_sizes

            features = self.faster_rcnn.backbone(images.tensors)
            proposals = self.faster_rcnn.rpn(images, features, targets)[0]

            if self.stage == "rpn":
                for boxes, original_image_size, transformed_image_size in zip(
                    proposals, original_image_sizes, transformed_image_sizes
                ):
                    boxes = box_ops.clip_boxes_to_image(boxes, transformed_image_size)

                    keep = box_ops.remove_small_boxes(boxes, min_size=self.min_size)
                    boxes = boxes[keep]

                    keep = box_ops.nms(boxes, self.pseudo_scores[: boxes.shape[0]], self.iou_threshold)
                    boxes = boxes[keep]

                    boxes = update_box_sizes_following_image_resize(boxes, transformed_image_size, original_image_size)

                    if self.tiling:
                        boxes = tile_boxes(boxes, original_image_size, self.tile_size)

                        keep = box_ops.nms(boxes, self.pseudo_scores[: boxes.shape[0]], self.tile_iou_threshold)
                        boxes = boxes[keep]

                    regions.append(boxes)
            elif self.stage == "rcnn":
                box_features = self.faster_rcnn.roi_heads.box_roi_pool(features, proposals, transformed_image_sizes)  # type: ignore
                box_features = self.faster_rcnn.roi_heads.box_head(box_features)  # type: ignore
                class_logits, box_regression = self.faster_rcnn.roi_heads.box_predictor(box_features)  # type: ignore
                box_predictions = self.post_process_box_predictions(
                    class_logits, box_regression, proposals, transformed_image_sizes
                )

                for boxes, original_image_size, transformed_image_size in zip(
                    box_predictions, original_image_sizes, transformed_image_sizes
                ):
                    boxes = update_box_sizes_following_image_resize(boxes, transformed_image_size, original_image_size)

                    if self.tiling:
                        boxes = tile_boxes(boxes, original_image_size, self.tile_size)

                        keep = box_ops.nms(boxes, self.pseudo_scores[: boxes.shape[0]], self.tile_iou_threshold)
                        boxes = boxes[keep]

                    regions.append(boxes)
            else:
                raise ValueError("Unknown stage {}".format(self.stage))

            return regions

    def post_process_box_predictions(
        self, class_logits: Tensor, box_regression: Tensor, proposals: List[Tensor], image_shapes: List[Tuple[int, int]]
    ) -> List[Tensor]:
        """Post-processes the box predictions.

        Args:
            class_logits (Tensor): Class logits.
            box_regression (Tensor): Box predictions of shape (N, 4).
            proposals (List[Tensor]): Proposals from the RPN.
            image_shapes (List[Tuple[int, int]]): Shapes of the transformed images.

        Returns:
            List[Tensor]: Post-processed box predictions of shape (N, 4).
        """
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.faster_rcnn.roi_heads.box_coder.decode(box_regression, proposals)  # type: ignore

        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)

        post_processed_boxes: List[Tensor] = []
        for boxes, scores, image_shape in zip(pred_boxes, pred_scores, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()

            # remove low scoring boxes
            keep = torch.nonzero(scores > self.rcnn_score_thresh).squeeze(1)
            boxes, scores = boxes[keep], scores[keep]

            # remove small boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=self.min_size)
            boxes, scores = boxes[keep], scores[keep]

            # non-maximum suppression, all boxes together
            keep = box_ops.nms(boxes, scores, self.iou_threshold)

            # keep only top-k scoring predictions
            keep = keep[: self.rcnn_detections_per_img]
            boxes = boxes[keep]

            post_processed_boxes.append(boxes)

        return post_processed_boxes


def tile_boxes(boxes: Tensor, image_size: Tuple[int, int], tile_size: int) -> Tensor:
    """Tile boxes.

    Args:
        boxes (Tensor): Box predictions, shape: [N, 4] - (x1, y1, x2, y2)
        image_size (Tuple[int, int]): Height and width of the image from which
            boxes are predicted.
        tile_size (int): Tile size.

    Returns:
        Tensor: Tiled boxes, shape: [N, 4] - (x1, y1, x2, y2)
    """
    assert tile_size % 2 == 0, "``tile_size`` must be power of 2."

    boxes = boxes.round_().to(torch.int32)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    x2[(widths % 2) == 1] += 1
    y2[(heights % 2) == 1] += 1

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    assert (widths % 2 == 0).all()
    assert (heights % 2 == 0).all()

    deltas = torch.zeros_like(boxes)
    deltas[:, 0] = (widths - tile_size) / 2
    deltas[:, 1] = (heights - tile_size) / 2
    deltas[:, 2] = (tile_size - widths) / 2
    deltas[:, 3] = (tile_size - heights) / 2

    boxes = boxes + deltas

    overhang = boxes[:, 0] * -1
    overhang_mask = overhang < 0
    overhang[overhang_mask] = 0
    boxes[:, 0] += overhang
    boxes[:, 2] += overhang

    overhang = boxes[:, 1] * -1
    overhang_mask = overhang < 0
    overhang[overhang_mask] = 0
    boxes[:, 1] += overhang
    boxes[:, 3] += overhang

    overhang = boxes[:, 2] - image_size[1] + 1
    overhang_mask = overhang < 0
    overhang[overhang_mask] = 0
    boxes[:, 0] -= overhang
    boxes[:, 2] -= overhang

    overhang = boxes[:, 3] - image_size[0] + 1
    overhang_mask = overhang < 0
    overhang[overhang_mask] = 0
    boxes[:, 1] -= overhang
    boxes[:, 3] -= overhang

    boxes = boxes.to(torch.float32)
    return boxes


def update_box_sizes_following_image_resize(boxes: Tensor, original_size: Size, new_size: Size) -> Tensor:
    """Update box sizes following image resize.

    Args:
        boxes (Tensor): Boxes of shape (N, 4) - (x1, y1, x2, y2).
        original_size (Size): Size of the original image.
        new_size (Size): Size of the transformed image.

    Returns:
        Tensor: Updated boxes of shape (N, 4) - (x1, y1, x2, y2).
    """
    height_ratio, width_ratio = (new_s / org_s for new_s, org_s in zip(new_size, original_size))
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * width_ratio
    xmax = xmax * width_ratio
    ymin = ymin * height_ratio
    ymax = ymax * height_ratio
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


def likelihood_to_class_threshold(likelihood: float) -> float:
    """Convert likelihood to class threshold.

    Args:
        likelihood (float): Input likelihood.

    Returns:
        float: Class threshold.
    """
    threshold = (torch.cos(torch.tensor(likelihood) * torch.pi / 2.0) ** 4.0).item()
    return threshold
