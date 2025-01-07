"""Helper functions for processing bounding box detections and annotations.

This module provides utility functions for converting between different bounding box
formats and handling bounding box operations.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from anomalib.utils.cv import connected_components_cpu, connected_components_gpu


def masks_to_boxes(
    masks: torch.Tensor,
    anomaly_maps: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Convert batch of segmentation masks to bounding box coordinates.

    Args:
        masks: Input tensor of masks. Can be one of:
            - shape ``(B, 1, H, W)``
            - shape ``(B, H, W)``
            - shape ``(H, W)``
        anomaly_maps: Optional anomaly maps. Can be one of:
            - shape ``(B, 1, H, W)``
            - shape ``(B, H, W)``
            - shape ``(H, W)``
            Used to determine anomaly scores for converted bounding boxes.

    Returns:
        Tuple containing:
            - List of length ``B`` where each element is tensor of shape ``(N, 4)``
              containing bounding box coordinates in ``xyxy`` format
            - List of length ``B`` where each element is tensor of length ``N``
              containing anomaly scores for each converted box

    Examples:
        >>> import torch
        >>> masks = torch.zeros((2, 1, 32, 32))
        >>> masks[0, 0, 10:20, 15:25] = 1  # Add box in first image
        >>> boxes, scores = masks_to_boxes(masks)
        >>> boxes[0]  # Coordinates for first image
        tensor([[15., 10., 24., 19.]])
    """
    height, width = masks.shape[-2:]
    # reshape to (B, 1, H, W) and cast to float
    masks = masks.view((-1, 1, height, width)).float()
    if anomaly_maps is not None:
        anomaly_maps = anomaly_maps.view((-1,) + masks.shape[-2:])

    if masks.is_cpu:
        batch_comps = connected_components_cpu(masks).squeeze(1)
    else:
        batch_comps = connected_components_gpu(masks).squeeze(1)

    batch_boxes = []
    batch_scores = []
    for im_idx, im_comps in enumerate(batch_comps):
        labels = torch.unique(im_comps)
        im_boxes = []
        im_scores = []
        for label in labels[labels != 0]:
            y_loc, x_loc = torch.where(im_comps == label)
            # add box
            box = torch.Tensor([torch.min(x_loc), torch.min(y_loc), torch.max(x_loc), torch.max(y_loc)]).to(
                masks.device,
            )
            im_boxes.append(box)
            if anomaly_maps is not None:
                im_scores.append(torch.max(anomaly_maps[im_idx, y_loc, x_loc]))
        batch_boxes.append(torch.stack(im_boxes) if im_boxes else torch.empty((0, 4), device=masks.device))
        batch_scores.append(torch.stack(im_scores) if im_scores else torch.empty(0, device=masks.device))

    return batch_boxes, batch_scores


def boxes_to_masks(boxes: list[torch.Tensor], image_size: tuple[int, int]) -> torch.Tensor:
    """Convert bounding boxes to segmentation masks.

    Args:
        boxes: List of length ``B`` where each element is tensor of shape ``(N, 4)``
            containing bounding box coordinates in ``xyxy`` format
        image_size: Output mask size as ``(H, W)``

    Returns:
        Binary masks of shape ``(B, H, W)`` where pixels contained within boxes
        are set to 1

    Examples:
        >>> boxes = [torch.tensor([[10, 15, 20, 25]])]  # One box in first image
        >>> masks = boxes_to_masks(boxes, (32, 32))
        >>> masks.shape
        torch.Size([1, 32, 32])
    """
    masks = torch.zeros((len(boxes), *image_size)).to(boxes[0].device)
    for im_idx, im_boxes in enumerate(boxes):
        for box in im_boxes:
            x_1, y_1, x_2, y_2 = box.int()
            masks[im_idx, y_1 : y_2 + 1, x_1 : x_2 + 1] = 1
    return masks


def boxes_to_anomaly_maps(boxes: torch.Tensor, scores: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    """Convert bounding boxes and scores to anomaly heatmaps.

    Args:
        boxes: List of length ``B`` where each element is tensor of shape ``(N, 4)``
            containing bounding box coordinates in ``xyxy`` format
        scores: List of length ``B`` where each element is 1D tensor of length ``N``
            containing anomaly scores for each box
        image_size: Output heatmap size as ``(H, W)``

    Returns:
        Anomaly heatmaps of shape ``(B, H, W)``. Pixels within each box are set to
        that box's anomaly score. For overlapping boxes, the highest score is used.

    Examples:
        >>> boxes = [torch.tensor([[10, 15, 20, 25]])]  # One box
        >>> scores = [torch.tensor([0.9])]  # Score for the box
        >>> maps = boxes_to_anomaly_maps(boxes, scores, (32, 32))
        >>> maps[0, 20, 15]  # Point inside box
        tensor(0.9000)
    """
    anomaly_maps = torch.zeros((len(boxes), *image_size)).to(boxes[0].device)
    for im_idx, (im_boxes, im_scores) in enumerate(zip(boxes, scores, strict=False)):
        im_map = torch.zeros((im_boxes.shape[0], *image_size))
        for box_idx, (box, score) in enumerate(zip(im_boxes, im_scores, strict=True)):
            x_1, y_1, x_2, y_2 = box.int()
            im_map[box_idx, y_1 : y_2 + 1, x_1 : x_2 + 1] = score
            anomaly_maps[im_idx], _ = im_map.max(dim=0)
    return anomaly_maps


def scale_boxes(boxes: torch.Tensor, image_size: torch.Size, new_size: torch.Size) -> torch.Tensor:
    """Scale bounding box coordinates to a new image size.

    Args:
        boxes: Boxes of shape ``(N, 4)`` in ``(x1, y1, x2, y2)`` format
        image_size: Original image size the boxes were computed for
        new_size: Target image size to scale boxes to

    Returns:
        Scaled boxes of shape ``(N, 4)`` in ``(x1, y1, x2, y2)`` format

    Examples:
        >>> boxes = torch.tensor([[10, 15, 20, 25]])
        >>> scaled = scale_boxes(boxes, (32, 32), (64, 64))
        >>> scaled
        tensor([[20., 30., 40., 50.]])
    """
    scale = torch.Tensor([*new_size]) / torch.Tensor([*image_size])
    return boxes * scale.repeat(2).to(boxes.device)
