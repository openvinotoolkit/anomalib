"""CDF normalizer."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pytorch_lightning.utilities.types import STEP_OUTPUT

from anomalib import trainer
from anomalib.post_processing.normalization import cdf
from anomalib.utils.metrics import AnomalyScoreDistribution

from .base import BaseNormalizer


class CDFNormalizer(BaseNormalizer):
    """CDF normalizer.

    Args:
        trainer (trainer.AnomalibTrainer): Trainer object.
    """

    def __init__(self, trainer: "trainer.AnomalibTrainer"):
        self.metric_class = AnomalyScoreDistribution
        super().__init__(trainer=trainer)

    def update(self, outputs: STEP_OUTPUT):
        """Applies CDF standardization to the batch.

        Args:
            outputs (STEP_OUTPUT): Outputs from the model.
        """
        stats = self.metric.to(outputs["pred_scores"].device)
        outputs["pred_scores"] = cdf.standardize(outputs["pred_scores"], stats.image_mean, stats.image_std)
        if "anomaly_maps" in outputs.keys():
            outputs["anomaly_maps"] = cdf.standardize(
                outputs["anomaly_maps"], stats.pixel_mean, stats.pixel_std, center_at=stats.image_mean
            )

    def normalize(self, outputs: STEP_OUTPUT):
        """Normalize the outputs.

        Args:
            outputs (STEP_OUTPUT): Outputs from the model.
        """
        assert self.trainer.image_threshold is not None, "Image threshold is not set"
        outputs["pred_scores"] = cdf.normalize(outputs["pred_scores"], self.trainer.image_threshold.value)
        if "anomaly_maps" in outputs.keys():
            assert self.trainer.pixel_threshold is not None, "Pixel threshold is not set"
            outputs["anomaly_maps"] = cdf.normalize(outputs["anomaly_maps"], self.trainer.pixel_threshold.value)
