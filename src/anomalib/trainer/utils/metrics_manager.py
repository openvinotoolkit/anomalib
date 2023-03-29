"""Assigns and updates metrics."""
from __future__ import annotations

import logging
from typing import List

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import EPOCH_OUTPUT

from anomalib.data import TaskType
from anomalib.models.components import AnomalyModule
from anomalib.utils.metrics import create_metric_collection
from anomalib.utils.metrics.collection import AnomalibMetricCollection

logger = logging.getLogger(__name__)


class MetricsManager:
    def __init__(self, image_metrics: list[str] | None = None, pixel_metrics: list[str] | None = None):
        super().__init__()
        self.image_metric_names = image_metrics
        self.pixel_metric_names = pixel_metrics

        self.image_metrics: AnomalibMetricCollection
        self.pixel_metrics: AnomalibMetricCollection

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

        self.image_metrics = create_metric_collection(image_metric_names, "image_")
        self.pixel_metrics = create_metric_collection(pixel_metric_names, "pixel_")

        self.image_metrics.set_threshold(anomaly_module.image_threshold.value)
        self.pixel_metrics.set_threshold(anomaly_module.pixel_threshold.value)

    def set_threshold(self, image_metrics_threshold: float = 0.5, pixel_metrics_threshold: float = 0.5) -> None:
        """Sets threshold.

        The default 0.5 value is used for testing.

        Args:
            image_metrics_threshold (float): Value to assign image metrics. Defaults to 0.5.
            pixel_metrics_threshold (float): Value to assign pixel metrics. Defaults to 0.5.
        """
        if self.image_metrics is not None:
            self.image_metrics.set_threshold(image_metrics_threshold)
        if self.pixel_metrics is not None:
            self.pixel_metrics.set_threshold(pixel_metrics_threshold)

    def _update_metrics(self, outputs: EPOCH_OUTPUT | List[EPOCH_OUTPUT]):
        """Update metrics.

        Args:
            outputs (EPOCH_OUTPUT | List[EPOCH_OUTPUT]): Step outputs
        """
        if isinstance(outputs, list):
            for output in outputs:
                self._update_metrics(output)
        else:
            self.image_metrics.cpu()
            self.image_metrics.update(outputs["pred_scores"], outputs["label"].int())
            if "mask" in outputs.keys() and "anomaly_maps" in outputs.keys():
                self.pixel_metrics.cpu()
                self.pixel_metrics.update(outputs["anomaly_maps"], outputs["mask"].int())

    def compute(self, outputs: EPOCH_OUTPUT | List[EPOCH_OUTPUT]) -> None:
        """Computes the metrics.

        1. Update metrics
        2. Compute metrics

        Args:
            outputs (EPOCH_OUTPUT): Epoch outputs
        """
        self._update_metrics(outputs)
        self.image_metrics.compute()
        self.pixel_metrics.compute()

    def log(self, trainer: Trainer, current_fx_name: str):
        """Log metrics.

        Args:
            trainer (Trainer): Lightning Trainer
            current_fx_name (str): Name of the current hook. Eg: ``validation_epoch_end``
        """
        if self.pixel_metrics.update_called:
            self._log_metrics(trainer, self.pixel_metrics, current_fx_name, prog_bar=True)
            self._log_metrics(trainer, self.image_metrics, current_fx_name, prog_bar=False)
        else:
            self._log_metrics(trainer, self.image_metrics, current_fx_name, prog_bar=True)

    def _log_metrics(
        self, trainer: Trainer, metrics: AnomalibMetricCollection, current_fx_name: str, prog_bar: bool = False
    ):
        """Log metrics to the trainer's result collection.

        Args:
            trainer (Trainer): Lightning Trainer
            metrics (AnomalibMetricCollection): Metrics to log
            current_fx_name (str): Name of the current hook. Eg: ``validation_epoch_end``
            prog_bar (bool, optional): Whether to log to the progress bar. Defaults to False.

        Raises:
            MisconfigurationException: _description_
        """
        if trainer._results is None:
            raise MisconfigurationException("Loop's result collection is not registered.")
        for key, value in metrics.items(keep_base=False, copy_state=False):
            trainer._results.log(current_fx_name, name=key, value=value, prog_bar=prog_bar)
