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
    def __init__(self, trainer: "trainer.AnomalibTrainer"):
        super().__init__(metric_class=MinMax, trainer=trainer)

    def update(self, outputs: STEP_OUTPUT):
        if "anomaly_maps" in outputs:
            self.metric(outputs["anomaly_maps"])
        elif "box_scores" in outputs:
            self.metric(torch.cat(outputs["box_scores"]))
        elif "pred_scores" in outputs:
            self.metric(outputs["pred_scores"])
        else:
            raise ValueError("No values found for normalization, provide anomaly maps, bbox scores, or image scores")

    def normalize(self, outputs: STEP_OUTPUT):
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
