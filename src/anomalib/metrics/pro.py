"""Implementation of PRO metric based on TorchMetrics.

This module provides the Per-Region Overlap (PRO) metric for evaluating anomaly
segmentation performance. The PRO metric computes the macro average of the
per-region overlap between predicted anomaly masks and ground truth masks.

Example:
    >>> import torch
    >>> from anomalib.metrics import PRO
    >>> # Create sample predictions and targets
    >>> preds = torch.rand(2, 1, 32, 32)  # Batch of 2 images
    >>> target = torch.zeros(2, 1, 32, 32)
    >>> target[0, 0, 10:20, 10:20] = 1  # Add anomalous region
    >>> # Initialize metric
    >>> pro = PRO()
    >>> # Update metric state
    >>> pro.update(preds, target)
    >>> # Compute PRO score
    >>> score = pro.compute()
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torchmetrics import Metric
from torchmetrics.functional import recall
from torchmetrics.utilities.data import dim_zero_cat

from anomalib.utils.cv import connected_components_cpu, connected_components_gpu

from .base import AnomalibMetric


class _PRO(Metric):
    """Per-Region Overlap (PRO) Score.

    This metric computes the macro average of the per-region overlap between the
    predicted anomaly masks and the ground truth masks. It first identifies
    connected components in the ground truth mask and then computes the overlap
    between each component and the predicted mask.

    Args:
        threshold (float, optional): Threshold used to binarize the predictions.
            Defaults to ``0.5``.
        kwargs: Additional arguments passed to the TorchMetrics base class.

    Attributes:
        target (list[torch.Tensor]): List storing ground truth masks from batches
        preds (list[torch.Tensor]): List storing predicted masks from batches
        threshold (float): Threshold for binarizing predictions

    Example:
        >>> import torch
        >>> from anomalib.metrics import PRO
        >>> # Create random predictions and targets
        >>> preds = torch.rand(2, 1, 32, 32)  # Batch of 2 images
        >>> target = torch.zeros(2, 1, 32, 32)
        >>> target[0, 0, 10:20, 10:20] = 1  # Add anomalous region
        >>> # Initialize and compute PRO score
        >>> pro = PRO(threshold=0.5)
        >>> pro.update(preds, target)
        >>> score = pro.compute()
    """

    target: list[torch.Tensor]
    preds: list[torch.Tensor]

    def __init__(self, threshold: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metric state with new predictions and targets.

        Args:
            predictions (torch.Tensor): Predicted anomaly masks of shape
                ``(B, 1, H, W)`` where B is batch size
            targets (torch.Tensor): Ground truth anomaly masks of shape
                ``(B, 1, H, W)``

        Example:
            >>> pro = PRO()
            >>> # Assuming preds and target are properly shaped tensors
            >>> pro.update(preds, target)
        """
        self.target.append(targets)
        self.preds.append(predictions)

    def compute(self) -> torch.Tensor:
        """Compute the macro average PRO score across all regions.

        Returns:
            torch.Tensor: Scalar tensor containing the PRO score averaged across
                all regions in all batches

        Example:
            >>> pro = PRO()
            >>> # After updating with several batches
            >>> score = pro.compute()
            >>> print(f"PRO Score: {score:.4f}")
        """
        target = dim_zero_cat(self.target)
        preds = dim_zero_cat(self.preds)

        # kornia expects N1HW format and float dtype
        target = target.unsqueeze(1).type(torch.float)
        comps = connected_components_gpu(target) if target.is_cuda else connected_components_cpu(target)
        return pro_score(preds, comps, threshold=self.threshold)


def pro_score(
    predictions: torch.Tensor,
    comps: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Calculate the PRO score for a batch of predictions.

    Args:
        predictions (torch.Tensor): Predicted anomaly masks of shape
            ``(B, 1, H, W)``
        comps (torch.Tensor): Labeled connected components of shape ``(B, H, W)``.
            Components should be labeled from 0 to N
        threshold (float, optional): Threshold for binarizing float predictions.
            Defaults to ``0.5``

    Returns:
        torch.Tensor: Scalar tensor containing the average PRO score

    Example:
        >>> # Assuming predictions and components are properly shaped tensors
        >>> score = pro_score(predictions, components, threshold=0.5)
        >>> print(f"PRO Score: {score:.4f}")
    """
    if predictions.dtype == torch.float:
        predictions = predictions > threshold

    n_comps = len(comps.unique())

    preds = comps.clone()
    # match the shapes in case one of the tensors is N1HW
    preds = preds.reshape(predictions.shape)
    preds[~predictions] = 0
    if n_comps == 1:  # only background
        return torch.Tensor([1.0])

    # Even though ignore_index is set to 0, the final average computed with
    # "macro" takes the entire length of the tensor into account. That's why we
    # need to manually subtract 1 from the number of components after taking the
    # sum
    recall_tensor = recall(
        preds.flatten(),
        comps.flatten(),
        task="multiclass",
        num_classes=n_comps,
        average=None,
        ignore_index=0,
    )
    return recall_tensor.sum() / (n_comps - 1)


class PRO(AnomalibMetric, _PRO):  # type: ignore[misc]
    """Wrapper to add AnomalibMetric functionality to PRO metric.

    This class inherits from both ``AnomalibMetric`` and ``_PRO`` to combine
    Anomalib's metric functionality with the PRO score computation.
    """
