"""Normalizer used in AnomalibTrainer.

This is responsible for setting up the normalization method.
"""

from __future__ import annotations

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics import Metric

import anomalib.trainer as core
from anomalib.models import AnomalyModule
from anomalib.post_processing import NormalizationMethod
from anomalib.post_processing.normalization import cdf, min_max
from anomalib.utils.metrics import get_normalization_metrics

# TODO separate cdf and minmax normalizer


class Normalizer:
    """The normalizer class is instantiated by the trainer.

    This is responsible for updating the normalization values and normalizing the outputs.

    Args:
        normalization_method (NormalizationMethod): Normalization method. Defaults to None
    """

    def __init__(
        self, trainer: core.AnomalibTrainer, normalization_method: NormalizationMethod = NormalizationMethod.NONE
    ):
        self.normalization_method: NormalizationMethod = normalization_method
        self.trainer = trainer

    @property
    def normalization_metrics(self) -> Metric | None:
        """Returns normalization metrics.

        Checks if the trainer has anomaly module. If the trainer does, it searches for the normalization metrics in the
        anomaly module. If it does not find it, it assigns the normalization metrics.
        """
        if self.trainer.lightning_module is not None:
            normalization_metrics = None
            if not hasattr(self.trainer.lightning_module, "normalization_metrics"):
                setattr(
                    self.trainer.lightning_module,
                    "normalization_metrics",
                    get_normalization_metrics(self.normalization_method),
                )
            if self.trainer.lightning_module.normalization_metrics is not None:
                normalization_metrics = self.trainer.lightning_module.normalization_metrics.cpu()
            return normalization_metrics
        else:
            raise ValueError("Trainer does not have a lightning module assigned.")

    def update_metrics(self, outputs: STEP_OUTPUT):
        """Update values

        Args:
            outputs (STEP_OUTPUT): Outputs used for gathering normalization metrics.
        """
        if self.normalization_metrics is None:
            return

        if self.normalization_method == NormalizationMethod.MIN_MAX:
            if "anomaly_maps" in outputs:
                self.normalization_metrics(outputs["anomaly_maps"])
            elif "box_scores" in outputs:
                self.normalization_metrics(torch.cat(outputs["box_scores"]))
            elif "pred_scores" in outputs:
                self.normalization_metrics(outputs["pred_scores"])
            else:
                raise ValueError(
                    "No values found for normalization, provide anomaly maps, bbox scores, or image scores"
                )
        elif self.normalization_method == NormalizationMethod.CDF:
            self._standardize_batch(outputs)

    def _standardize_batch(self, outputs: STEP_OUTPUT) -> None:
        """Only used by CDF normalization"""
        if self.normalization_metrics is None:
            return
        stats = self.normalization_metrics.to(outputs["pred_scores"].device)
        outputs["pred_scores"] = cdf.standardize(outputs["pred_scores"], stats.image_mean, stats.image_std)
        if "anomaly_maps" in outputs.keys():
            outputs["anomaly_maps"] = cdf.standardize(
                outputs["anomaly_maps"], stats.pixel_mean, stats.pixel_std, center_at=stats.image_mean
            )

    def _normalize_batch(self, outputs: STEP_OUTPUT, anomaly_module: AnomalyModule):
        """Normalize the batch.

        Args:
            outputs (STEP_OUTPUT): Output of the batch.
            anomaly_module (AnomalyModule): Anomaly module.
        """
        if self.normalization_metrics is None:
            return
        image_threshold = anomaly_module.image_threshold.value.cpu()
        pixel_threshold = anomaly_module.pixel_threshold.value.cpu()
        if self.normalization_method == NormalizationMethod.MIN_MAX:
            outputs["pred_scores"] = min_max.normalize(
                outputs["pred_scores"], image_threshold, self.normalization_metrics.min, self.normalization_metrics.max
            )
            if "anomaly_maps" in outputs:
                outputs["anomaly_maps"] = min_max.normalize(
                    outputs["anomaly_maps"],
                    pixel_threshold,
                    self.normalization_metrics.min,
                    self.normalization_metrics.max,
                )
            if "box_scores" in outputs:
                outputs["box_scores"] = [
                    min_max.normalize(
                        scores, pixel_threshold, self.normalization_metrics.min, self.normalization_metrics.max
                    )
                    for scores in outputs["box_scores"]
                ]
        elif self.normalization_method == NormalizationMethod.CDF:
            outputs["pred_scores"] = cdf.normalize(outputs["pred_scores"], anomaly_module.image_threshold.value)
            if "anomaly_maps" in outputs.keys():
                outputs["anomaly_maps"] = cdf.normalize(outputs["anomaly_maps"], anomaly_module.pixel_threshold.value)

    def normalize(self, anomaly_module: AnomalyModule, outputs: STEP_OUTPUT) -> None:
        """Normalize the outputs.

        Args:
            anomaly_module (AnomalyModule): Anomaly Module
            outputs (STEP_OUTPUT): outputs to normalize
        """
        if self.normalization_metrics is None:
            return

        if self.normalization_method == NormalizationMethod.MIN_MAX:
            self._normalize_batch(outputs, anomaly_module)
        elif self.normalization_method == NormalizationMethod.CDF:
            self._standardize_batch(outputs)
            self._normalize_batch(outputs, anomaly_module)
