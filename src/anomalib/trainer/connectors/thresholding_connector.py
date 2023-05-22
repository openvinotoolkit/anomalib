"""Assigns and updates thresholds."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from importlib import import_module

from pytorch_lightning.utilities.types import STEP_OUTPUT

from anomalib import trainer
from anomalib.data import TaskType
from anomalib.utils.metrics import BaseAnomalyThreshold, F1AdaptiveThreshold


class ThresholdingConnector:
    """Computes and updates thresholds.

    Used in AnomalibTrainer.

    Args:
        trainer (trainer.AnomalibTrainer): Trainer object
        image_threshold_method (dice | None): Thresholding method. If None, adaptive thresholding is used.
        pixel_threshold_method (dice | None): Thresholding method. If None, adaptive thresholding is used.
    """

    def __init__(
        self,
        trainer: "trainer.AnomalibTrainer",
        image_threshold_method: dict | None = None,  # TODO change this in the CLI
        pixel_threshold_method: dict | None = None,
    ) -> None:
        self.image_threshold_method = image_threshold_method
        self.pixel_threshold_method = pixel_threshold_method
        self.image_threshold: BaseAnomalyThreshold
        self.pixel_threshold: BaseAnomalyThreshold
        self.trainer = trainer

    def initialize(self) -> None:
        """Assigns pixel and image thresholds to the Anomalib trainer."""
        # Private members are accessed because we don't want to set the thresholds if they are already set
        if not hasattr(self, "image_threshold"):
            self.image_threshold = self._get_threshold_metric(self.image_threshold_method)
        if not hasattr(self, "pixel_threshold"):
            self.pixel_threshold = self._get_threshold_metric(self.pixel_threshold_method)

    def compute(self):
        """Compute thresholds.

        Args:
            outputs (EPOCH_OUTPUT | List[EPOCH_OUTPUT]): Outputs are only used to check if the model has pixel level
                predictions.
        """
        if self.image_threshold is not None:
            self.image_threshold.compute()
        if self.trainer.task_type in (TaskType.SEGMENTATION, TaskType.DETECTION):
            self.pixel_threshold.compute()
        else:
            self.pixel_threshold.value = self.image_threshold.value

    def update(self, outputs: STEP_OUTPUT) -> None:
        """updates adaptive threshold in case thresholding type is ADAPTIVE.

        Args:
            outputs (STEP_OUTPUT): Step outputs.
        """
        self.image_threshold.cpu()
        self.image_threshold.update(outputs["pred_scores"], outputs["label"].int())
        if (
            self.trainer.task_type != TaskType.CLASSIFICATION
            and "anomaly_maps" in outputs.keys()
            and "mask" in outputs.keys()
        ):
            self.pixel_threshold.cpu()
            # TODO this should use bounding boxes for detection task type
            self.pixel_threshold.update(outputs["anomaly_maps"], outputs["mask"].int())

    def _get_threshold_metric(self, threshold_method: dict | None) -> BaseAnomalyThreshold:
        """Get instantiated threshold metric.

        Args:
            threshold_method (dict | None): Threshold method. Defaults to F1AdaptiveThreshold.

        Returns:
            Instantiated threshold metric.
        """
        # TODO change this in the CLI
        threshold_metric: BaseAnomalyThreshold
        if threshold_method is None:
            threshold_metric = F1AdaptiveThreshold()
        else:
            _class_path = threshold_method.get("class_path", "F1AdaptiveThreshold")
            try:
                if len(_class_path.split(".")) > 1:  # When the entire class path is provided
                    threshold_module = import_module(".".join(_class_path.split(".")[:-1]))
                    _threshold_class = getattr(threshold_module, _class_path.split(".")[-1])
                else:
                    threshold_module = import_module("anomalib.utils.metrics.thresholding")
                    _threshold_class = getattr(threshold_module, _class_path)
            except (AttributeError, ModuleNotFoundError) as exception:
                raise ValueError(f"Threshold class {_class_path} not found") from exception

            init_args = threshold_method.get("init_args")
            init_args = init_args if init_args is not None else {}
            threshold_metric = _threshold_class(**init_args)

        return threshold_metric
