"""Assigns and updates thresholds."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from warnings import warn

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

from anomalib import trainer
from anomalib.data import TaskType
from anomalib.post_processing import ThresholdMethod
from anomalib.utils.metrics import AnomalyScoreThreshold


class ThresholdingConnector:
    """Computes and updates thresholds.

    Used in AnomalibTrainer.

    Args:
        trainer (trainer.AnomalibTrainer): Trainer object
        threshold_method (ThresholdMethod): Thresholding method to use. Defaults to ``ThresholdMethod.ADAPTIVE``.
        manual_image_threshold (Optional[float]): Image threshold in case manual threshold is used. Defaults to None.
        manual_pixel_threshold (Optional[float]) = Pixel threshold in case manual threshold is used. Defaults to None.
    """

    def __init__(
        self,
        trainer: "trainer.AnomalibTrainer",
        threshold_method: ThresholdMethod = ThresholdMethod.ADAPTIVE,
        manual_image_threshold: float | None = None,
        manual_pixel_threshold: float | None = None,
    ) -> None:
        if threshold_method == ThresholdMethod.ADAPTIVE and all(
            i is not None for i in (manual_image_threshold, manual_pixel_threshold)
        ):
            warn(
                "When `threshold_method` is set to `adaptive`, `manual_image_threshold` and `manual_pixel_threshold` "
                "must not be set. Ignoring manual thresholds."
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

    def initialize(self) -> None:
        """Assigns pixel and image thresholds to the model.

        This allows us to export the metrics along with the torch model.
        """

        if self.threshold_method == ThresholdMethod.MANUAL:
            self.trainer.pixel_threshold.value = torch.tensor(self.manual_pixel_threshold).cpu()
            self.trainer.image_threshold.value = torch.tensor(self.manual_image_threshold).cpu()

    def compute(self):
        """Compute thresholds.

        Args:
            outputs (EPOCH_OUTPUT | List[EPOCH_OUTPUT]): Outputs are only used to check if the model has pixel level
                predictions.
        """
        if self.trainer.image_threshold is not None:
            self.trainer.image_threshold.compute()
        if self.trainer.task_type in (TaskType.SEGMENTATION, TaskType.DETECTION):
            self.trainer.pixel_threshold.compute()
        else:
            self.trainer.pixel_threshold.value = self.trainer.image_threshold.value

    def update(self, outputs: STEP_OUTPUT) -> None:
        """updates adaptive threshold in case thresholding type is ADAPTIVE.

        Args:
            outputs (STEP_OUTPUT): Step outputs.
        """
        if self.threshold_method == ThresholdMethod.ADAPTIVE:
            self._update_thresholds(self.trainer.image_threshold, self.trainer.pixel_threshold, outputs)

    @staticmethod
    def _update_thresholds(
        image_metric: AnomalyScoreThreshold,
        pixel_metric: AnomalyScoreThreshold,
        outputs: STEP_OUTPUT,
    ) -> None:
        image_metric.cpu()
        image_metric.update(outputs["pred_scores"], outputs["label"].int())
        if "mask" in outputs.keys() and "anomaly_maps" in outputs.keys():
            pixel_metric.cpu()
            pixel_metric.update(outputs["anomaly_maps"], outputs["mask"].int())
