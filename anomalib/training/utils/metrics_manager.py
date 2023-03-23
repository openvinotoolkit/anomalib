"""Assigns and updates metrics."""
from __future__ import annotations

import logging
from typing import List

from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT

from anomalib.data import TaskType
from anomalib.models.components import AnomalyModule
from anomalib.utils.metrics import create_metric_collection

logger = logging.getLogger(__name__)


class MetricsManager:
    def __init__(self, image_metrics: list[str] | None = None, pixel_metrics: list[str] | None = None):
        super().__init__()
        self.image_metric_names = image_metrics
        self.pixel_metric_names = pixel_metrics

    def setup(self, anomaly_module: AnomalyModule, task: TaskType) -> None:
        """Setup image and pixel-level AnomalibMetricsCollection within Anomalib Model.

        Args:
            anomaly_module (AnomalyModule): Anomalib Model that inherits pl LightningModule.
            task (TaskType): Task type
        """

        image_metric_names = [] if self.image_metric_names is None else self.image_metric_names

        pixel_metric_names: list[str]
        if self.pixel_metric_names is None:
            pixel_metric_names = []
        elif task == TaskType.CLASSIFICATION:
            pixel_metric_names = []
            logger.warning(
                "Cannot perform pixel-level evaluation when task type is classification. "
                "Ignoring the following pixel-level metrics: %s",
                self.pixel_metric_names,
            )
        else:
            pixel_metric_names = self.pixel_metric_names

        anomaly_module.image_metrics = create_metric_collection(image_metric_names, "image_")
        anomaly_module.pixel_metrics = create_metric_collection(pixel_metric_names, "pixel_")

        anomaly_module.image_metrics.set_threshold(anomaly_module.image_threshold.value)
        anomaly_module.pixel_metrics.set_threshold(anomaly_module.pixel_threshold.value)

    def update_metrics(self, anomaly_module: AnomalyModule, outputs: EPOCH_OUTPUT | List[EPOCH_OUTPUT] | STEP_OUTPUT):
        if isinstance(outputs, list):
            for output in outputs:
                self.update_metrics(anomaly_module, output)
        else:
            anomaly_module.image_metrics.cpu()
            anomaly_module.image_metrics.update(outputs["pred_scores"], outputs["label"].int())
            if "mask" in outputs.keys() and "anomaly_maps" in outputs.keys():
                anomaly_module.pixel_metrics.cpu()
                anomaly_module.pixel_metrics.update(outputs["anomaly_maps"], outputs["mask"].int())
