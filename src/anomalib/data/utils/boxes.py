"""Helper functions for processing bounding box detections and annotations."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from anomalib.utils.cv import connected_components_cpu, connected_components_gpu


def masks_to_boxes(
    masks: torch.Tensor,
    anomaly_maps: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Convert a batch of segmentation masks to bounding box coordinates.

    Args:
        masks (torch.Tensor): Input tensor of shape (B, 1, H, W), (B, H, W) or (H, W)
        anomaly_maps (Tensor | None, optional): Anomaly maps of shape (B, 1, H, W), (B, H, W) or (H, W) which are
            used to determine an anomaly score for the converted bounding boxes.

    Returns:
        list[torch.Tensor]: A list of length B where each element is a tensor of shape (N, 4)
            containing the bounding box coordinates of the objects in the masks in xyxy format.
        list[torch.Tensor]: A list of length B where each element is a tensor of length (N)
            containing an anomaly score for each of the converted boxes.
    """
    height, width = masks.shape[-2:]
    masks = masks.view((-1, 1, height, width)).float()  # reshape to (B, 1, H, W) and cast to float
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
    """Convert bounding boxes to segmentations masks.

    Args:
        boxes (list[torch.Tensor]): A list of length B where each element is a tensor of shape (N, 4)
            containing the bounding box coordinates of the regions of interest in xyxy format.
        image_size (tuple[int, int]): Image size of the output masks in (H, W) format.

    Returns:
        Tensor: torch.Tensor of shape (B, H, W) in which each slice is a binary mask showing the pixels contained by a
            bounding box.
    """
    masks = torch.zeros((len(boxes), *image_size)).to(boxes[0].device)
    for im_idx, im_boxes in enumerate(boxes):
        for box in im_boxes:
            x_1, y_1, x_2, y_2 = box.int()
            masks[im_idx, y_1 : y_2 + 1, x_1 : x_2 + 1] = 1
    return masks


def boxes_to_anomaly_maps(boxes: torch.Tensor, scores: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    """Convert bounding box coordinates to anomaly heatmaps.

    Args:
        boxes (list[torch.Tensor]): A list of length B where each element is a tensor of shape (N, 4)
            containing the bounding box coordinates of the regions of interest in xyxy format.
        scores (list[torch.Tensor]): A list of length B where each element is a 1D tensor of length N
            containing the anomaly scores for each region of interest.
        image_size (tuple[int, int]): Image size of the output masks in (H, W) format.

    Returns:
        Tensor: torch.Tensor of shape (B, H, W). The pixel locations within each bounding box are collectively
            assigned the anomaly score of the bounding box. In the case of overlapping bounding boxes,
            the highest score is used.
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
    """Scale bbox coordinates to a new image size.

    Args:
        boxes (torch.Tensor): Boxes of shape (N, 4) - (x1, y1, x2, y2).
        image_size (Size): Size of the original image in which the bbox coordinates were retrieved.
        new_size (Size): New image size to which the bbox coordinates will be scaled.

    Returns:
        Tensor: Updated boxes of shape (N, 4) - (x1, y1, x2, y2).
    """
    scale = torch.Tensor([*new_size]) / torch.Tensor([*image_size])
    return boxes * scale.repeat(2).to(boxes.device)
