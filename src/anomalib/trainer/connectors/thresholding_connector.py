"""Assigns and updates thresholds."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from torchmetrics.metric import Metric

from anomalib.data import TaskType
from anomalib.post_processing import ThresholdMethod


class ThresholdingConnector:
    """Computes and updates thresholds.

    Used in AnomalibTrainer.

    Args:
        trainer (trainer.AnomalibTrainer): Trainer object
        threshold_method (ThresholdMethod): Thresholding method to use. Defaults to ``ThresholdMethod.ADAPTIVE``.
    """

    def __init__(
        self,
        pixel_threshold: None | Metric = None,
        image_threshold: None | Metric = None,
        threshold_method: ThresholdMethod = ThresholdMethod.ADAPTIVE,
        task_type: TaskType = TaskType.SEGMENTATION,
    ) -> None:
        self.threshold_method = threshold_method
        self.pixel_threshold = pixel_threshold
        self.image_threshold = image_threshold
        self.task_type = task_type

    def initialize(self) -> None:
        """Assigns pixel and image thresholds to the model.

        This allows us to export the metrics along with the torch model.
        """

        # if self.threshold_method == ThresholdMethod.MANUAL:
        #     self.trainer.pixel_threshold.value = torch.tensor(self.manual_pixel_threshold).cpu()
        #     self.trainer.image_threshold.value = torch.tensor(self.manual_image_threshold).cpu()
        pass

    def compute(self):
        """Compute thresholds.

        Args:
            outputs (EPOCH_OUTPUT | List[EPOCH_OUTPUT]): Outputs are only used to check if the model has pixel level
                predictions.
        """
        if self.threshold_method == ThresholdMethod.ADAPTIVE:
            if self.image_threshold is not None:
                self.image_threshold.compute()
            if self.task_type in (TaskType.SEGMENTATION, TaskType.DETECTION):
                self.pixel_threshold.compute()
            else:
                self.pixel_threshold.value = self.image_threshold.value

    def update(self, outputs) -> None:
        """updates adaptive threshold in case thresholding type is ADAPTIVE.

        Args:
            outputs (STEP_OUTPUT): Step outputs.
        """
        if self.threshold_method == ThresholdMethod.ADAPTIVE:
            self.image_threshold.cpu()
            self.image_threshold.update(outputs["pred_scores"], outputs["label"].int())
            if "mask" in outputs.keys() and "anomaly_maps" in outputs.keys():
                self.pixel_threshold.cpu()
                self.pixel_threshold.update(outputs["anomaly_maps"], outputs["mask"].int())
