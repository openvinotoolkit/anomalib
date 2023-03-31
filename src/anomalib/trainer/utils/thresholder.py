"""Assigns and updates thresholds."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import List

import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT

import anomalib.trainer as core
from anomalib.data import TaskType
from anomalib.models import AnomalyModule
from anomalib.post_processing import ThresholdMethod
from anomalib.utils.metrics import AnomalyScoreThreshold, get_thresholding_metrics


class Thresholder:
    """Computes and updates thresholds.

    Used in AnomalibTrainer.

    Args:
        trainer (core.AnomalibTrainer): Trainer object
        threshold_method (ThresholdMethod): Thresholding method to use. Defaults to ``ThresholdMethod.ADAPTIVE``.
        manual_image_threshold (Optional[float]): Image threshold in case manual threshold is used. Defaults to None.
        manual_pixel_threshold (Optional[float]) = Pixel threshold in case manual threshold is used. Defaults to None.
    """

    def __init__(
        self,
        trainer: core.AnomalibTrainer,
        threshold_method: ThresholdMethod = ThresholdMethod.ADAPTIVE,
        manual_image_threshold: float | None = None,
        manual_pixel_threshold: float | None = None,
    ) -> None:
        if threshold_method == ThresholdMethod.ADAPTIVE and all(
            i is not None for i in (manual_image_threshold, manual_pixel_threshold)
        ):
            raise ValueError(
                "When `threshold_method` is set to `adaptive`, `manual_image_threshold` and `manual_pixel_threshold` "
                "must not be set."
            )

        if threshold_method == ThresholdMethod.MANUAL and all(
            i is None for i in (manual_image_threshold, manual_pixel_threshold)
        ):
            raise ValueError(
                "When `threshold_method` is set to `manual`, `manual_image_threshold` and `manual_pixel_threshold` "
                "must be set."
            )

        self.threshold_method = threshold_method
        self.manual_image_threshold = manual_image_threshold
        self.manual_pixel_threshold = manual_pixel_threshold
        self.trainer = trainer

    @property
    def anomaly_module(self) -> AnomalyModule:
        """Returns anomaly module.

        We can't directly access the anomaly module in ``__init__`` because it is not available till it is passed to the
        trainer.
        """
        return self.trainer.lightning_module

    def initialize(self) -> None:
        """Assigns pixel and image thresholds to the model.

        This allows us to export the metrics along with the torch model.
        """
        if not hasattr(self.anomaly_module, "pixel_threshold"):
            self.anomaly_module.pixel_threshold = get_thresholding_metrics()
        if not hasattr(self.anomaly_module, "image_threshold"):
            self.anomaly_module.image_threshold = get_thresholding_metrics()

        if self.threshold_method == ThresholdMethod.MANUAL:
            self.anomaly_module.pixel_threshold.value = torch.tensor(self.manual_pixel_threshold).cpu()
            self.anomaly_module.image_threshold.value = torch.tensor(self.manual_image_threshold).cpu()

    def compute(self):
        """Compute thresholds.

        Args:
            outputs (EPOCH_OUTPUT | List[EPOCH_OUTPUT]): Outputs are only used to check if the model has pixel level
                predictions.
        """
        if self.anomaly_module.image_threshold is not None:
            self.anomaly_module.image_threshold.compute()
        if self.trainer.task_type in (TaskType.SEGMENTATION, TaskType.DETECTION):
            self.anomaly_module.pixel_threshold.compute()
        else:
            self.anomaly_module.pixel_threshold.value = self.anomaly_module.image_threshold.value

    def update(self, outputs: STEP_OUTPUT) -> None:
        """updates adaptive threshold in case thresholding type is ADAPTIVE.

        Args:
            outputs (STEP_OUTPUT): Step outputs.
        """
        if self.threshold_method == ThresholdMethod.ADAPTIVE:
            self._update_thresholds(self.anomaly_module.image_threshold, self.anomaly_module.pixel_threshold, outputs)

    @staticmethod
    def _update_thresholds(
        image_metric: AnomalyScoreThreshold,
        pixel_metric: AnomalyScoreThreshold,
        outputs: EPOCH_OUTPUT | List[EPOCH_OUTPUT] | STEP_OUTPUT,
    ) -> None:
        if isinstance(outputs, list):
            for output in outputs:
                Thresholder._update_thresholds(image_metric, pixel_metric, output)
        else:
            image_metric.cpu()
            image_metric.update(outputs["pred_scores"], outputs["label"].int())
            if "mask" in outputs.keys() and "anomaly_maps" in outputs.keys():
                pixel_metric.cpu()
                pixel_metric.update(outputs["anomaly_maps"], outputs["mask"].int())
