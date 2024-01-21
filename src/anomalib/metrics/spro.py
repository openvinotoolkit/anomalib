"""Implementation of sPRO metric based on TorchMetrics."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import torch
from torchmetrics import Metric
from torchmetrics.functional import recall
from torchmetrics.utilities.data import dim_zero_cat

from anomalib.utils.cv import connected_components_cpu, connected_components_gpu

logger = logging.getLogger(__name__)

class sPRO(Metric):
    """Saturated Per-Region Overlap (sPRO) Score.

    This metric computes the macro average of the saturated per-region overlap between the
    predicted anomaly masks and the ground truth masks.

    Args:
        threshold (float): Threshold used to binarize the predictions.
            Defaults to ``0.5``.
        kwargs: Additional arguments to the TorchMetrics base class.

    Example:
        Import the metric from the package:

        >>> import torch
        >>> from anomalib.metrics import sPRO

        Create random ``preds`` and ``labels`` tensors:

        >>> labels = torch.randint(low=0, high=2, size=(1, 10, 5), dtype=torch.float32)
        >>> preds = torch.rand_like(labels)

        Compute the sPRO score for labels and preds:

        >>> spro = sPRO(threshold=0.5)
        >>> spro.update(preds, _, labels)
        >>> spro.compute()
        tensor(0.6333)

        .. note::
            Note that the example above shows random predictions and labels.
            Therefore, the sPRO score above may not be reproducible.

    """

    targets: list[torch.Tensor]
    preds: list[torch.Tensor]
    saturation_config: dict

    def __init__(self, threshold: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold
        self.saturation_config = {}
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, predictions: torch.Tensor, _: torch.Tensor, masks: torch.Tensor) -> None:
        """Compute the sPRO score for the current batch.

        Args:
            predictions (torch.Tensor): Predicted anomaly masks
            _ (torch.Tensor): Unused argument, but needed for different metrics within the same AnomalibMetricCollection
            masks (torch.Tensor): Ground truth anomaly masks with non-binary values and, original height and width

        Example:
            To update the metric state for the current batch, use the ``update`` method:

            >>> spro.update(preds, _, labels)
        """
        assert masks is not None
        self.targets.append(masks)
        self.preds.append(predictions)

    def compute(self) -> torch.Tensor:
        """Compute the macro average of the sPRO score across all masks in all batches.

        Example:
            To compute the metric based on the state accumulated from multiple batches, use the ``compute`` method:

            >>> spro.compute()
            tensor(0.5433)
        """
        targets = dim_zero_cat(self.targets)
        preds = dim_zero_cat(self.preds)
        return spro_score(preds, targets, threshold=self.threshold, saturation_config=self.saturation_config)

def spro_score(predictions: torch.Tensor, targets: torch.Tensor,
                threshold: float = 0.5, saturation_config: dict = {}) -> torch.Tensor:
    """Calculate the sPRO score for a batch of predictions.

    Args:
        predictions (torch.Tensor): Predicted anomaly masks
        targets: (torch.Tensor): Ground truth anomaly masks with non-binary values and, original height and width
        threshold (float): When predictions are passed as float, the threshold is used to binarize the predictions.
        saturation_config (dict): Saturations configuration for each label (pixel value) as the keys

    Returns:
        torch.Tensor: Scalar value representing the average sPRO score for the input batch.
    """
    predictions = torch.nn.functional.interpolate(predictions.unsqueeze(1), targets.shape[1:])

    # Apply threshold to binary predictions
    if predictions.dtype == torch.float:
        predictions = predictions > threshold

    score = torch.tensor(0.0)

    # Iterate for each image in the batch
    for i, target in enumerate(targets):
        unique_labels = torch.unique(target)

        # Iterate for each ground-truth mask per image
        for label in unique_labels[1:]:
            # Calculate true positive
            target_per_label = target == label
            true_pos = torch.sum(predictions[i] & target_per_label)

            # Calculate the areas of the ground-truth
            defect_areas = torch.sum(target_per_label)

            if len(saturation_config) > 0:
                # Adjust saturation threshold based on configuration
                saturation_per_label = saturation_config[label.int().item()]
                saturation_threshold = torch.minimum(torch.tensor(saturation_per_label["saturation_threshold"]), defect_areas)
                if saturation_per_label["relative_saturation"]:
                    saturation_threshold *= defect_areas
            else:
                # Handle case when saturation_config is empty
                logger.warning("The saturation_config attribute is empty, the threshold is set to the defect areas."
                               "This is equivalent to PRO metric but with the 'region' are separated by mask files")
                saturation_threshold = defect_areas

            # Update score with minimum of true_pos/saturation_threshold and 1.0
            score += torch.minimum(true_pos / saturation_threshold, torch.tensor(1.0))

    # Calculate the mean score
    return torch.mean(score)
