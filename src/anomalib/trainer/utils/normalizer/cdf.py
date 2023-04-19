"""CDF normalizer."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pytorch_lightning.utilities.types import STEP_OUTPUT

import anomalib.trainer as trainer
from anomalib.post_processing.normalization import cdf
from anomalib.utils.metrics import AnomalyScoreDistribution

from .base import BaseNormalizer


class CDFNormalizer(BaseNormalizer):
    def __init__(self, trainer: "trainer.AnomalibTrainer"):
        super().__init__(metric_class=AnomalyScoreDistribution, trainer=trainer)

    def update(self, outputs: STEP_OUTPUT):
        self._standardize_batch(outputs)

    def _standardize_batch(self, outputs: STEP_OUTPUT) -> None:
        """Only used by CDF normalization"""
        stats = self.metric.to(outputs["pred_scores"].device)
        outputs["pred_scores"] = cdf.standardize(outputs["pred_scores"], stats.image_mean, stats.image_std)
        if "anomaly_maps" in outputs.keys():
            outputs["anomaly_maps"] = cdf.standardize(
                outputs["anomaly_maps"], stats.pixel_mean, stats.pixel_std, center_at=stats.image_mean
            )

    def normalize(self, outputs: STEP_OUTPUT):
        outputs["pred_scores"] = cdf.normalize(outputs["pred_scores"], self.anomaly_module.image_threshold.value)
        if "anomaly_maps" in outputs.keys():
            outputs["anomaly_maps"] = cdf.normalize(outputs["anomaly_maps"], self.anomaly_module.pixel_threshold.value)
