# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Input tiling and geometric augmentation for GRD-Net."""

from collections.abc import Sequence

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2.functional import rotate

from anomalib.data.utils.tiler import Tiler


def tile_image_and_roi(
    images: torch.Tensor,
    roi_masks: torch.Tensor,
    tile_size: int | Sequence[int] = (128, 128),
    stride: int | Sequence[int] = (64, 64),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract spatially aligned image and ROI tiles.

    Images and ROI masks are concatenated before tiling so that both tensors use
    exactly the same tile geometry and ordering.

    Args:
        images: RGB image batch with shape ``[B, 3, H, W]``.
        roi_masks: ROI mask batch with shape ``[B, 1, H, W]``.
        tile_size: Tile height and width.
        stride: Vertical and horizontal tile stride.

    Returns:
        Tuple containing image tiles and binary ROI tiles.

    Raises:
        ValueError: If the image and ROI shapes are incompatible.
    """
    _validate_image_and_roi(images, roi_masks)
    roi_masks = roi_masks.to(device=images.device, dtype=images.dtype)
    combined = torch.cat((images, roi_masks), dim=1)
    tiles = Tiler(tile_size=tile_size, stride=stride).tile(combined)
    image_tiles, roi_tiles = tiles[:, :3], tiles[:, 3:]
    return image_tiles, (roi_tiles > 0.5).to(dtype=images.dtype)


def rotate_image_and_roi(
    image_tiles: torch.Tensor,
    roi_tiles: torch.Tensor,
    angles: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate each image and ROI tile pair by the same angle.

    Args:
        image_tiles: RGB tiles with shape ``[N, 3, H, W]``.
        roi_tiles: ROI tiles with shape ``[N, 1, H, W]``.
        angles: Optional rotation angles with shape ``[N]``. When omitted,
            angles are sampled uniformly from ``[-90, 90]`` degrees.

    Returns:
        Tuple containing rotated image and binary ROI tiles.

    Raises:
        ValueError: If the tile shapes or supplied angles are incompatible.
    """
    _validate_image_and_roi(image_tiles, roi_tiles)
    if angles is None:
        angles = torch.empty(image_tiles.shape[0], device=image_tiles.device).uniform_(-90.0, 90.0)
    if angles.ndim != 1 or angles.shape[0] != image_tiles.shape[0]:
        msg = f"Expected one rotation angle per tile, got shape {tuple(angles.shape)}."
        raise ValueError(msg)

    rotated_images = []
    rotated_rois = []
    for image, roi_mask, angle in zip(image_tiles, roi_tiles, angles, strict=True):
        rotated_images.append(
            rotate(image, angle=float(angle.item()), interpolation=InterpolationMode.BILINEAR, fill=0),
        )
        rotated_rois.append(
            rotate(roi_mask, angle=float(angle.item()), interpolation=InterpolationMode.NEAREST, fill=0),
        )

    images = torch.stack(rotated_images)
    rois = torch.stack(rotated_rois)
    return images, (rois > 0.5).to(dtype=image_tiles.dtype)


def _validate_image_and_roi(images: torch.Tensor, roi_masks: torch.Tensor) -> None:
    """Validate paired RGB images and single-channel ROI masks."""
    if images.ndim != 4 or images.shape[1] != 3:
        msg = f"Expected RGB images with shape [B, 3, H, W], got {tuple(images.shape)}."
        raise ValueError(msg)
    if roi_masks.ndim != 4 or roi_masks.shape[1] != 1:
        msg = f"Expected ROI masks with shape [B, 1, H, W], got {tuple(roi_masks.shape)}."
        raise ValueError(msg)
    if images.shape[0] != roi_masks.shape[0] or images.shape[-2:] != roi_masks.shape[-2:]:
        msg = (
            "Image and ROI batch/spatial dimensions must match, "
            f"got {tuple(images.shape)} and {tuple(roi_masks.shape)}."
        )
        raise ValueError(msg)
