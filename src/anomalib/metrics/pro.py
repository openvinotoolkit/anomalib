"""Implementation of PRO metric based on TorchMetrics."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torchmetrics import Metric
from torchmetrics.functional import recall
from torchmetrics.utilities.data import dim_zero_cat

from anomalib.utils.cv import connected_components_cpu, connected_components_gpu


class PRO(Metric):
    """Per-Region Overlap (PRO) Score.

    This metric computes the macro average of the per-region overlap between the
    predicted anomaly masks and the ground truth masks.

    Args:
        threshold (float): Threshold used to binarize the predictions.
            Defaults to ``0.5``.
        kwargs: Additional arguments to the TorchMetrics base class.

    Example:
        Import the metric from the package:

        >>> import torch
        >>> from anomalib.metrics import PRO

        Create random ``preds`` and ``labels`` tensors:

        >>> labels = torch.randint(low=0, high=2, size=(1, 10, 5), dtype=torch.float32)
        >>> preds = torch.rand_like(labels)

        Compute the PRO score for labels and preds:

        >>> pro = PRO(threshold=0.5)
        >>> pro.update(preds, labels)
        >>> pro.compute()
        tensor(0.5433)

        .. note::
            Note that the example above shows random predictions and labels.
            Therefore, the PRO score above may not be reproducible.

    """

    target: list[torch.Tensor]
    preds: list[torch.Tensor]

    def __init__(self, threshold: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Compute the PRO score for the current batch.

        Args:
            predictions (torch.Tensor): Predicted anomaly masks (Bx1xHxW)
            targets (torch.Tensor): Ground truth anomaly masks (Bx1xHxW)

        Example:
            To update the metric state for the current batch, use the ``update`` method:

            >>> pro.update(preds, labels)
        """
        self.target.append(targets)
        self.preds.append(predictions)

    def compute(self) -> torch.Tensor:
        """Compute the macro average of the PRO score across all regions in all batches.

        Example:
            To compute the metric based on the state accumulated from multiple batches, use the ``compute`` method:

            >>> pro.compute()
            tensor(0.5433)
        """
        target = dim_zero_cat(self.target)
        preds = dim_zero_cat(self.preds)

        target = target.unsqueeze(1).type(torch.float)  # kornia expects N1HW and FloatTensor format
        comps = connected_components_gpu(target) if target.is_cuda else connected_components_cpu(target)
        return pro_score(preds, comps, threshold=self.threshold)


def pro_score(predictions: torch.Tensor, comps: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Calculate the PRO score for a batch of predictions.

    Args:
        predictions (torch.Tensor): Predicted anomaly masks (Bx1xHxW)
        comps: (torch.Tensor): Labeled connected components (BxHxW). The components should be labeled from 0 to N
        threshold (float): When predictions are passed as float, the threshold is used to binarize the predictions.

    Returns:
        torch.Tensor: Scalar value representing the average PRO score for the input batch.
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

    # Even though ignore_index is set to 0, the final average computed with "macro"
    # takes the entire length of the tensor into account. That's why we need to manually
    # subtract 1 from the number of components after taking the sum
    recall_tensor = recall(
        preds.flatten(),
        comps.flatten(),
        task="multiclass",
        num_classes=n_comps,
        average=None,
        ignore_index=0,
    )
    return recall_tensor.sum() / (n_comps - 1)
