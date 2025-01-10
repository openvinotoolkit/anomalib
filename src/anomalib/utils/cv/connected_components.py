"""Connected component labeling for anomaly detection.

This module provides functions for performing connected component labeling on both
GPU and CPU. Connected components are used to identify and label contiguous
regions in binary images, which is useful for post-processing anomaly detection
results.

Example:
    >>> import torch
    >>> from anomalib.utils.cv import connected_components_gpu
    >>> # Create a binary mask tensor (1 for anomaly, 0 for normal)
    >>> mask = torch.zeros(1, 1, 4, 4)
    >>> mask[0, 0, 1:3, 1:3] = 1  # Create a 2x2 square anomaly
    >>> # Get labeled components
    >>> labels = connected_components_gpu(mask)
    >>> print(labels.unique())  # Should show [0, 1] for background and one component
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import cv2
import numpy as np
import torch
from kornia.contrib import connected_components


def connected_components_gpu(image: torch.Tensor, num_iterations: int = 1000) -> torch.Tensor:
    """Perform connected component labeling on GPU.

    Labels connected regions in a binary image and remaps the labels sequentially
    from 0 to N, where N is the number of unique components. Uses the GPU for
    faster processing of large images.

    Args:
        image (torch.Tensor): Binary input image tensor of shape ``(B, 1, H, W)``
            where ``B`` is batch size, ``H`` is height and ``W`` is width.
            Values should be binary (0 or 1).
        num_iterations (int, optional): Number of iterations for the connected
            components algorithm. Higher values may be needed for complex regions.
            Defaults to 1000.

    Returns:
        torch.Tensor: Integer tensor of same shape as input, containing labeled
            components from 0 to N. Background (zero) pixels in the input remain
            ``0``, while connected regions are labeled with integers from ``1``
            to ``N``.

    Example:
        >>> import torch
        >>> from anomalib.utils.cv import connected_components_gpu
        >>> # Create a binary mask with a 2x2 square anomaly
        >>> mask = torch.zeros(1, 1, 4, 4)
        >>> mask[0, 0, 1:3, 1:3] = 1
        >>> labels = connected_components_gpu(mask)
        >>> print(labels.unique())  # Should show tensor([0, 1])
        >>> print(labels[0, 0])  # Show the labeled components
        tensor([[0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0]])
    """
    components = connected_components(image, num_iterations=num_iterations)

    # remap component values from 0 to N
    labels = components.unique()
    for new_label, old_label in enumerate(labels):
        components[components == old_label] = new_label

    return components.int()


def connected_components_cpu(image: torch.Tensor) -> torch.Tensor:
    """Perform connected component labeling on CPU.

    Labels connected regions in a binary image using OpenCV's implementation.
    Ensures unique labeling across batched inputs by remapping component labels
    sequentially.

    Args:
        image (torch.Tensor): Binary input tensor of shape ``(B, 1, H, W)`` where
            ``B`` is batch size, ``H`` is height and ``W`` is width. Values should
            be binary (``0`` or ``1``).

    Returns:
        torch.Tensor: Integer tensor of same shape as input, containing labeled
            components from ``0`` to ``N``. Background (zero) pixels in the input
            remain ``0``, while connected regions are labeled with integers from
            ``1`` to ``N``, ensuring unique labels across the batch.

    Example:
        >>> import torch
        >>> from anomalib.utils.cv import connected_components_cpu
        >>> # Create a binary mask with a 2x2 square anomaly
        >>> mask = torch.zeros(1, 1, 4, 4)
        >>> mask[0, 0, 1:3, 1:3] = 1
        >>> labels = connected_components_cpu(mask)
        >>> print(labels.unique())  # Should show tensor([0, 1])
        >>> print(labels[0, 0])  # Show the labeled components
        tensor([[0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0]])

    Note:
        This function uses OpenCV's ``connectedComponents`` implementation which
        runs on CPU. For GPU acceleration, use :func:`connected_components_gpu`
        instead.
    """
    components = torch.zeros_like(image)
    label_idx = 1
    for i, msk in enumerate(image):
        mask = msk.squeeze().cpu().numpy().astype(np.uint8)
        _, comps = cv2.connectedComponents(mask)
        # remap component values to make sure every component has a unique value when outputs are concatenated
        for label in np.unique(comps)[1:]:
            components[i, 0, ...][np.where(comps == label)] = label_idx
            label_idx += 1
    return components.int()
