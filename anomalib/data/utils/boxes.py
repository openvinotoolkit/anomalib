"""Helper functions for processing bounding box detections and annotations."""

from typing import List, Tuple

import torch
from torch import Tensor

from anomalib.utils.cv import connected_components_cpu, connected_components_gpu


def masks_to_boxes(masks: Tensor) -> List[Tensor]:
    """Convert a batch of segmentation masks to bounding box coordinates.

    Args:
        masks (Tensor): Input tensor of shape (B, 1, H, W), (B, H, W) or (H, W)

    Returns:
        List[Tensor]: A list of length B where each element is a tensor of shape (N, 4) containing the bounding box
            coordinates of the objects in the masks in xyxy format.
    """
    masks = masks.view((-1, 1) + masks.shape[-2:])  # reshape to (B, 1, H, W)
    masks = masks.float()

    if masks.is_cuda:
        batch_comps = connected_components_gpu(masks).squeeze(1)
    else:
        batch_comps = connected_components_cpu(masks).squeeze(1)

    batch_boxes = []
    for im_comps in batch_comps:
        labels = torch.unique(im_comps)
        im_boxes = []
        for label in labels[labels != 0]:
            y_loc, x_loc = torch.where(im_comps == label)
            im_boxes.append(Tensor([torch.min(x_loc), torch.min(y_loc), torch.max(x_loc), torch.max(y_loc)]))
        batch_boxes.append(torch.stack(im_boxes) if len(im_boxes) > 0 else torch.empty((0, 4)))
    return batch_boxes


def boxes_to_masks(boxes: List[Tensor], image_size: Tuple[int, int]) -> Tensor:
    """Convert bounding boxes to segmentations masks.

    Args:
        boxes (List[Tensor]): A list of length B where each element is a tensor of shape (N, 4) containing the bounding
            box coordinates of the regions of interest in xyxy format.
        image_size (Tuple[int, int]): Image size of the output masks in (H, W) format.

    Returns:
        Tensor: Tensor of shape (B, H, W) in which each slice is a binary mask showing the pixels contained by a
            bounding box.
    """
    masks = torch.zeros((len(boxes),) + image_size)
    for im_idx, im_boxes in enumerate(boxes):
        for box in im_boxes:
            x_1, y_1, x_2, y_2 = box.int()
            masks[im_idx, y_1:y_2, x_1:x_2] = 1
    return masks


def boxes_to_anomaly_maps(boxes: Tensor, scores: Tensor, image_size: Tuple[int, int]) -> Tensor:
    """Convert bounding box coordinates to anomaly heatmaps.

    Args:
        boxes (List[Tensor]): A list of length B where each element is a tensor of shape (N, 4) containing the bounding
            box coordinates of the regions of interest in xyxy format.
        scores (List[Tensor]): A list of length B where each element is a 1D tensor of length N containing the anomaly
            scores for each region of interest.
        image_size (Tuple[int, int]): Image size of the output masks in (H, W) format.

    Returns:
        Tensor: Tensor of shape (B, H, W). The pixel locations within each bounding box are collectively assigned the
            anomaly score of the bounding box. In the case of overlapping bounding boxes, the highest score is used.
    """
    anomaly_maps = torch.zeros((len(boxes),) + image_size).to(boxes[0].device)
    for im_idx, (im_boxes, im_scores) in enumerate(zip(boxes, scores)):
        im_map = torch.zeros((im_boxes.shape[0],) + image_size)
        for box_idx, (box, score) in enumerate(zip(im_boxes, im_scores)):
            x_1, y_1, x_2, y_2 = box.int()
            im_map[box_idx, y_1:y_2, x_1:x_2] = score
            anomaly_maps[im_idx], _ = im_map.max(dim=0)
    return anomaly_maps
