"""Implementation of PRO metric based on TorchMetrics."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List

import cv2
import numpy as np
import torch
from kornia.contrib import connected_components
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import recall
from torchmetrics.utilities.data import dim_zero_cat


class PRO(Metric):
    """Per-Region Overlap (PRO) Score."""

    target: List[Tensor]
    preds: List[Tensor]

    def __init__(self, threshold: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Compute the PRO score for the current batch."""

        self.target.append(targets)
        self.preds.append(predictions)

    def compute(self) -> Tensor:
        """Compute the macro average of the PRO score across all regions in all batches."""
        target = dim_zero_cat(self.target)
        preds = dim_zero_cat(self.preds)

        if target.is_cuda:
            comps = connected_components_gpu(target.unsqueeze(1))
        else:
            comps = connected_components_cpu(target.unsqueeze(1))
        pro = pro_score(preds, comps, threshold=self.threshold)
        return pro


def pro_score(predictions: Tensor, comps: Tensor, threshold: float = 0.5) -> Tensor:
    """Calculate the PRO score for a batch of predictions.

    Args:
        predictions (Tensor): Predicted anomaly masks (Bx1xHxW)
        comps: (Tensor): Labeled connected components (BxHxW). The components should be labeled from 0 to N
        threshold (float): When predictions are passed as float, the threshold is used to binarize the predictions.

    Returns:
        Tensor: Scalar value representing the average PRO score for the input batch.
    """
    if predictions.dtype == torch.float:
        predictions = predictions > threshold

    n_comps = len(comps.unique())

    preds = comps.clone()
    preds[~predictions] = 0
    if n_comps == 1:  # only background
        return torch.Tensor([1.0])
    pro = recall(preds.flatten(), comps.flatten(), num_classes=n_comps, average="macro", ignore_index=0)
    return pro


def connected_components_gpu(binary_input: Tensor, num_iterations: int = 1000) -> Tensor:
    """Perform connected component labeling on GPU and remap the labels from 0 to N.

    Args:
        binary_input (Tensor): Binary input data from which we want to extract connected components (Bx1xHxW)
        num_iterations (int): Number of iterations used in the connected component computation.

    Returns:
        Tensor: Components labeled from 0 to N.
    """
    components = connected_components(binary_input, num_iterations=num_iterations)

    # remap component values from 0 to N
    labels = components.unique()
    for new_label, old_label in enumerate(labels):
        components[components == old_label] = new_label

    return components.int()


def connected_components_cpu(image: Tensor) -> Tensor:
    """Connected component labeling on CPU.

    Args:
        image (Tensor): Binary input data from which we want to extract connected components (Bx1xHxW)

    Returns:
        Tensor: Components labeled from 0 to N.
    """
    components = torch.zeros_like(image)
    label_idx = 1
    for i, mask in enumerate(image):
        mask = mask.squeeze().numpy().astype(np.uint8)
        _, comps = cv2.connectedComponents(mask)
        # remap component values to make sure every component has a unique value when outputs are concatenated
        for label in np.unique(comps)[1:]:
            components[i, 0, ...][np.where(comps == label)] = label_idx
            label_idx += 1
    return components.int()
