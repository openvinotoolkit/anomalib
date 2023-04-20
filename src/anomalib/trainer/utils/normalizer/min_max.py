"""MinMax normalizer."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

import anomalib.trainer as trainer
from anomalib.post_processing.normalization import min_max
from anomalib.utils.metrics import MinMax

from .base import BaseNormalizer


class MinMaxNormalizer(BaseNormalizer):
    """MinMax normalizer.

    Args:
        trainer (trainer.AnomalibTrainer): Trainer object.
    """

    def __init__(self, trainer: "trainer.AnomalibTrainer"):
        self.metric_class = MinMax
        super().__init__(trainer=trainer)

    def update(self, outputs: STEP_OUTPUT):
        """Update the min and max values based on the batch.

        Args:
            outputs (STEP_OUTPUT): Outputs from the model.

        Raises:
            ValueError: If no values are found for normalization.
        """
        if "anomaly_maps" in outputs:
            self.metric(outputs["anomaly_maps"])
        elif "box_scores" in outputs:
            self.metric(torch.cat(outputs["box_scores"]))
        elif "pred_scores" in outputs:
            self.metric(outputs["pred_scores"])
        else:
            raise ValueError("No values found for normalization, provide anomaly maps, bbox scores, or image scores")

    def normalize(self, outputs: STEP_OUTPUT):
        """Normalize the outputs.

        Args:
            outputs (STEP_OUTPUT): Outputs from the model.
        """
        outputs["pred_scores"] = min_max.normalize(
            outputs["pred_scores"], self.anomaly_module.image_threshold.value.cpu(), self.metric.min, self.metric.max
        )
        if "anomaly_maps" in outputs:
            outputs["anomaly_maps"] = min_max.normalize(
                outputs["anomaly_maps"],
                self.anomaly_module.pixel_threshold.value.cpu(),
                self.metric.min,
                self.metric.max,
            )
        if "box_scores" in outputs:
            outputs["box_scores"] = [
                min_max.normalize(
                    scores, self.anomaly_module.pixel_threshold.value.cpu(), self.metric.min, self.metric.max
                )
                for scores in outputs["box_scores"]
            ]
