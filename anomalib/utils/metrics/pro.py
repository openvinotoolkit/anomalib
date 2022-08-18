"""Implementation of PRO metric based on TorchMetrics."""
import warnings
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
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

        self.add_state("preds", default=[], dist_reduce_fx="cat")  # pylint: disable=not-callable
        self.add_state("target", default=[], dist_reduce_fx="cat")  # pylint: disable=not-callable

    def update(self, predictions: Tensor, targets: Tensor) -> None:  # type: ignore  # pylint: disable=arguments-differ
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


def connected_components_gpu(binary_input: torch.Tensor, max_iterations: int = 1000) -> Tensor:
    """Pytorch implementation for Connected Component Labeling on GPU.

    Args:
        binary_input (Tensor): Binary input data from which we want to extract connected components (Bx1xHxW)
        max_iterations (int): Maximum number of iterations used in the connected component computaion.

    Returns:
        Tensor: Components labeled from 0 to N.
    """
    mask = binary_input.bool()

    batch, _, height, width = binary_input.shape
    components = torch.arange(batch * height * width, device=binary_input.device, dtype=torch.float).reshape(
        (batch, 1, height, width)
    )
    components[~mask] = 0

    converged = False
    for _ in range(max_iterations):
        previous = components.clone()
        components[mask] = F.max_pool2d(components, kernel_size=3, stride=1, padding=1)[mask]
        if torch.all(torch.eq(components, previous)):
            converged = True
            break

    if not converged:
        warnings.warn(
            f"Max iterations ({max_iterations}) reached before converging. Connected component results may be "
            f"inaccurate. Consider increasing the maximum number of iterations."
        )

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
    components_list = []
    label_idx = 1
    for mask in image:
        mask = mask.squeeze().numpy().astype(np.uint8)
        _, comps = cv2.connectedComponents(mask)
        # remap component values to make sure every component has a unique value when outputs are concatenated
        for label in np.unique(comps)[1:]:
            comps[np.where(comps == label)] = label_idx
            label_idx += 1
        components_list.append(comps)
    components = torch.Tensor(np.stack(components_list)).unsqueeze(1).int()
    return components
