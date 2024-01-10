"""Connected component labeling."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import cv2
import numpy as np
import torch
from kornia.contrib import connected_components


def connected_components_gpu(image: torch.Tensor, num_iterations: int = 1000) -> torch.Tensor:
    """Perform connected component labeling on GPU and remap the labels from 0 to N.

    Args:
        image (torch.Tensor): Binary input image from which we want to extract connected components (Bx1xHxW)
        num_iterations (int): Number of iterations used in the connected component computation.

    Returns:
        Tensor: Components labeled from 0 to N.
    """
    components = connected_components(image, num_iterations=num_iterations)

    # remap component values from 0 to N
    labels = components.unique()
    for new_label, old_label in enumerate(labels):
        components[components == old_label] = new_label

    return components.int()


def connected_components_cpu(image: torch.Tensor) -> torch.Tensor:
    """Perform connected component labeling on CPU.

    Args:
        image (torch.Tensor): Binary input data from which we want to extract connected components (Bx1xHxW)

    Returns:
        Tensor: Components labeled from 0 to N.
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
