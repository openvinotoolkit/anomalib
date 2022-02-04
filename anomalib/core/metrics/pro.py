"""Implementation of PRO metric based on TorchMetrics."""
import warnings
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import recall


class PRO(Metric):
    """Per-Region Overlap (PRO) Score."""

    def __init__(self, threshold: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        if not torch.cuda.is_available():
            warnings.warn(
                "Computation of the PRO metric is optimized for the GPU, but cuda is not available on your device. "
                "Because of this, the PRO computation will significantly slow down code execution."
            )
        self.threshold = threshold

        self.add_state("pro", default=torch.tensor(0.0), dist_reduce_fx="sum")  # pylint: disable=not-callable
        self.add_state("n_regions", default=torch.tensor(0), dist_reduce_fx="sum")  # pylint: disable=not-callable
        self.pro: Tensor
        self.n_regions: Tensor

    def update(self, predictions: Tensor, targets: Tensor) -> None:  # type: ignore  # pylint: disable=arguments-differ
        """Compute the PRO score for the current batch."""
        if torch.cuda.is_available():
            predictions = predictions.cuda()
            targets = targets.cuda()

        comps, n_comps = connected_components(targets.unsqueeze(1))
        pro = pro_score(predictions, comps, threshold=self.threshold)

        self.pro += pro * (n_comps - 1)
        self.n_regions += n_comps - 1

    def compute(self) -> Tensor:
        """Compute the macro average of the PRO score across all regions in all batches."""
        return self.pro / self.n_regions


def pro_score(predictions: Tensor, comps: Tensor, threshold: float = 0.5) -> Tensor:
    """Calculate the PRO score for a batch of predictions.

    Args:
        predictions (Tensor): Predicted anomaly masks (Bx1xHxW)
        comps: (Tensor): Labeled connected components (BxHxW)
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


def connected_components(binary_input: torch.Tensor, max_iterations: int = 500) -> Tuple[torch.Tensor, int]:
    """Pytorch implementation for Connected Component Labeling.

    Adapted from https://gist.github.com/efirdc/5d8bd66859e574c683a504a4690ae8bc

    Args:
        binary_input (Tensor): Binary input data from which we want to extract connected components (Bx1xHxW)
        max_iterations (int): Maximum number of iterations used in the connected component computaion.

    Returns:
        Tensor: Components labeled from 0 to N.
        int: number of connected components that were identified.
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

    # remap component values
    labels = components.unique()
    for new_label, old_label in enumerate(labels):
        components[components == old_label] = new_label

    return components.int(), len(labels)
