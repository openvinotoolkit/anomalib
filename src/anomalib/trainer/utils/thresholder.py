"""Assigns and updates thresholds."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from importlib import import_module

from pytorch_lightning.utilities.types import STEP_OUTPUT

from anomalib import trainer
from anomalib.data import TaskType
from anomalib.utils.metrics import AdaptiveScoreThreshold, BaseAnomalyScoreThreshold


class Thresholder:
    """Computes and updates thresholds.

    Used in AnomalibTrainer.

    Args:
        trainer (trainer.AnomalibTrainer): Trainer object
        image_threshold_method (BaseAnomalyScoreThreshold): Thresholding method. If None, adaptive thresholding is used.
        pixel_threshold_method (BaseAnomalyScoreThreshold): Thresholding method. If None, adaptive thresholding is used.
    """

    def __init__(
        self,
        trainer: trainer.AnomalibTrainer,
        image_threshold_method: dict | None = None,
        pixel_threshold_method: dict | None = None,
    ) -> None:
        self.image_threshold_method = image_threshold_method
        self.pixel_threshold_method = pixel_threshold_method
        self.trainer = trainer

    def initialize(self) -> None:
        """Assigns pixel and image thresholds to the model.

        This allows us to export the metrics along with the torch model.
        """

        self.setup("image_threshold", self.image_threshold_method)
        self.setup("pixel_threshold", self.pixel_threshold_method)

    def setup(self, attribute: str, threshold_method: dict | None):
        """Setup thresholds.

        Args:
            attribute (str): Property to setup.
        """
        if not hasattr(self.trainer, attribute) or getattr(self.trainer, attribute) is None:
            setattr(self.trainer, attribute, self._get_thresholder(threshold_method))

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
        image_metric = self.trainer.image_threshold
        pixel_metric = self.trainer.pixel_threshold
        image_metric.cpu()
        image_metric.update(outputs["pred_scores"], outputs["label"].int())
        if (
            self.trainer.task_type != TaskType.CLASSIFICATION
            and "anomaly_maps" in outputs.keys()
            and "mask" in outputs.keys()
        ):
            pixel_metric.cpu()
            # TODO this should use bounding boxes for detection task type
            pixel_metric.update(outputs["anomaly_maps"], outputs["mask"].int())

    def _get_thresholder(self, threshold_method: dict | None) -> BaseAnomalyScoreThreshold:
        """Get threshold method.

        Args:
            threshold_method (dict | None): Threshold method. Defaults to AdaptiveScoreThreshold.

        Returns:
            Instantiated threshold method.
        """
        thresholder: BaseAnomalyScoreThreshold
        if threshold_method is None:
            thresholder = AdaptiveScoreThreshold()
        else:
            _class_path = threshold_method.get("class_path", "AdaptiveScoreThreshold")
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
            thresholder = _threshold_class(**init_args)

        return thresholder
