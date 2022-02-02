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
        self.threshold = threshold
        self.scores: list = []  # average pro score per batch
        self.n_regions: list = []  # number of regions found in each batch

    def update(self, predictions: Tensor, targets: Tensor) -> None:  # type: ignore  # pylint: disable=arguments-differ
        """Compute the PRO score for the current batch."""
        if predictions.dtype == torch.float:
            predictions = predictions > self.threshold
        comps, n_comps = connected_components(targets.unsqueeze(1))
        preds = comps.clone()
        preds[~predictions] = 0
        pro = recall(preds.flatten(), comps.flatten(), num_classes=n_comps, average="macro", ignore_index=0)
        self.scores.append(pro)
        self.n_regions.append(n_comps)

    def compute(self) -> Tensor:
        """Compute the macro average of the PRO score across all regions in all batches."""
        pro = torch.dot(Tensor(self.scores), Tensor(self.n_regions)) / sum(self.n_regions)
        return pro


def connected_components(binary_input: torch.Tensor, max_iterations: int = 500) -> Tuple[torch.Tensor, int]:
    """Pytorch implementation for Connected Component Labeling.

    Adapted from https://gist.github.com/efirdc/5d8bd66859e574c683a504a4690ae8bc

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
