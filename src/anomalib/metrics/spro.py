"""Implementation of SPRO metric based on TorchMetrics."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torchmetrics import Metric

from anomalib.data.utils import validate_path

logger = logging.getLogger(__name__)


class SPRO(Metric):
    """Saturated Per-Region Overlap (SPRO) Score.

    This metric computes the macro average of the saturated per-region overlap between the
    predicted anomaly masks and the ground truth masks.

    Args:
        threshold (float): Threshold used to binarize the predictions.
            Defaults to ``0.5``.
        saturation_config (str | Path): Path to the saturation configuration file.
            Defaults: ``None`` (which the score is equivalent to PRO metric, but with the 'region' are
            separated by mask files.
        kwargs: Additional arguments to the TorchMetrics base class.

    Example:
        Import the metric from the package:

        >>> import torch
        >>> from anomalib.metrics import SPRO

        Create random ``preds`` and ``labels`` tensors:

        >>> labels = torch.randint(low=0, high=2, size=(2, 10, 5), dtype=torch.float32)
        >>> labels = [labels]
        >>> preds = torch.rand_like(labels[0][:1])

        Compute the SPRO score for labels and preds:

        >>> spro = SPRO(threshold=0.5)
        >>> spro.update(preds, labels)
        >>> spro.compute()
        tensor(0.6333)

        .. note::
            Note that the example above shows random predictions and labels.
            Therefore, the SPRO score above may not be reproducible.

    """

    def __init__(self, threshold: float = 0.5, saturation_config: str | Path | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold
        self.saturation_config = load_saturation_config(saturation_config) if saturation_config is not None else None
        if self.saturation_config is None:
            logger.warning(
                "The saturation_config attribute is empty, the threshold is set to the defect area."
                "This is equivalent to PRO metric but with the 'region' are separated by mask files",
            )
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: torch.Tensor, masks: list[torch.Tensor]) -> None:
        """Compute the SPRO score for the current batch.

        Args:
            predictions (torch.Tensor): Predicted anomaly masks.
            masks (list[torch.Tensor]): Ground truth anomaly masks with original height and width. Each element in the
                list is a tensor list of masks for the corresponding image.

        Example:
            To update the metric state for the current batch, use the ``update`` method:

            >>> spro.update(preds, labels)
        """
        score, total = spro_score(
            predictions=predictions,
            targets=masks,
            threshold=self.threshold,
            saturation_config=self.saturation_config,
        )
        self.score += score
        self.total += total

    def compute(self) -> torch.Tensor:
        """Compute the macro average of the SPRO score across all masks in all batches.

        Example:
            To compute the metric based on the state accumulated from multiple batches, use the ``compute`` method:

            >>> spro.compute()
            tensor(0.5433)
        """
        if self.total == 0:  # only background/normal images
            return torch.Tensor([1.0])
        return self.score / self.total


def spro_score(
    predictions: torch.Tensor,
    targets: list[torch.Tensor],
    threshold: float = 0.5,
    saturation_config: dict | None = None,
) -> torch.Tensor:
    """Calculate the SPRO score for a batch of predictions.

    Args:
        predictions (torch.Tensor): Predicted anomaly masks.
        targets: (list[torch.Tensor]): Ground truth anomaly masks with original height and width. Each element in the
            list is a tensor list of masks for the corresponding image.
        threshold (float): When predictions are passed as float, the threshold is used to binarize the predictions.
        saturation_config (dict): Saturations configuration for each label (pixel value) as the keys.
            Defaults: ``None`` (which the score is equivalent to PRO metric, but with the 'region' are
            separated by mask files.

    Returns:
        torch.Tensor: Scalar value representing the average SPRO score for the input batch.
    """
    # Add batch dim if not exist
    if len(predictions.shape) == 2:
        predictions = predictions.unsqueeze(0)

    # Resize the prediction to have the same size as the target mask
    predictions = torch.nn.functional.interpolate(predictions.unsqueeze(1), targets[0].shape[-2:])

    # Apply threshold to binary predictions
    if predictions.dtype == torch.float:
        predictions = predictions > threshold

    score = torch.tensor(0.0)
    total = 0
    # Iterate for each image in the batch
    for i, target in enumerate(targets):
        # Iterate for each ground-truth mask per image
        for mask in target:
            label = torch.max(mask)
            if label == 0:  # Skip if only normal/background
                continue
            # Calculate true positive
            target_per_label = mask == label
            true_pos = torch.sum(predictions[i] & target_per_label)

            # Calculate the anomalous area of the ground-truth
            defect_area = torch.sum(target_per_label)

            if saturation_config is not None:
                # Adjust saturation threshold based on configuration
                saturation_per_label = saturation_config[label.int().item()]
                saturation_threshold = saturation_per_label["saturation_threshold"]

                if saturation_per_label["relative_saturation"]:
                    saturation_threshold *= defect_area

                # Check if threshold is larger than defect area
                if saturation_threshold > defect_area:
                    warning_msg = (
                        f"Saturation threshold for label {label.int().item()} is larger than defect area. "
                        "Setting it to defect area."
                    )
                    logger.warning(warning_msg)
                    saturation_threshold = defect_area
            else:
                # Handle case when saturation_config is empty
                saturation_threshold = defect_area

            # Update score with minimum of true_pos/saturation_threshold and 1.0
            score += torch.minimum(true_pos / saturation_threshold, torch.tensor(1.0))
            total += 1
    return score, total


def load_saturation_config(config_path: str | Path) -> dict[int, Any] | None:
    """Load saturation configurations from a JSON file.

    Args:
        config_path (str | Path): Path to the saturation configuration file.

    Returns:
        Dict | None: A dictionary with pixel values as keys and the corresponding configurations as values.
            Return None if the config file is not found.

    Example JSON format in the config file of MVTec LOCO dataset:
    [
        {
            "defect_name": "1_additional_pushpin",
            "pixel_value": 255,
            "saturation_threshold": 6300,
            "relative_saturation": false
        },
        {
            "defect_name": "2_additional_pushpins",
            "pixel_value": 254,
            "saturation_threshold": 12600,
            "relative_saturation": false
        },
        ...
    ]
    """
    try:
        config_path = validate_path(config_path)
        with Path.open(config_path) as file:
            configs = json.load(file)
        # Create a dictionary with pixel values as keys
        return {conf["pixel_value"]: conf for conf in configs}
    except FileNotFoundError:
        logger.warning("The saturation config file %s does not exist. Returning None.", config_path)
        return None
