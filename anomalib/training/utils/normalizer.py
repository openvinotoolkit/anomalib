"""Normalizer used in AnomalibTrainer.

This is responsible for setting up the normalization method.
"""

from __future__ import annotations

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics import Metric

from anomalib.models import AnomalyModule
from anomalib.post_processing import NormalizationMethod
from anomalib.post_processing.normalization import cdf, min_max
from anomalib.utils.metrics import AnomalyScoreDistribution, MinMax

# TODO separate cdf and minmax normalizer


class Normalizer:
    """The normalizer class is instantiated by the trainer.

    This is responsible for updating the normalization values and normalizing the outputs.

    Args:
        normalization_method (NormalizationMethod): Normalization method. Defaults to None
    """

    def __init__(self, normalization_method: NormalizationMethod = NormalizationMethod.NONE):
        self.normalization_method = normalization_method
        self.normalization_metrics: Metric

    def setup(self):
        """Setup normalization metrics."""
        # TODO change logic here
        if not hasattr(self, "normalization_metrics"):
            if self.normalization_method == NormalizationMethod.MIN_MAX:
                self.normalization_metrics = MinMax().cpu()
            elif self.normalization_method == NormalizationMethod.CDF:
                # TODO CDF only works for padim and stfpm. Check condition
                # TODO throw error if nncf optimization is enabled.
                self.normalization_metrics = AnomalyScoreDistribution().cpu()
            else:
                raise ValueError(f"Normalization method {self.normalization_method} is not supported.")

    def update_metrics(self, outputs: STEP_OUTPUT):
        """Update values

        Args:
            outputs (STEP_OUTPUT): Outputs used for gathering normalization metrics.
        """
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
        stats = self.normalization_metrics.to(outputs["pred_scores"].device)
        outputs["pred_scores"] = cdf.standardize(outputs["pred_scores"], stats.image_mean, stats.image_std)
        if "anomaly_maps" in outputs.keys():
            outputs["anomaly_maps"] = cdf.standardize(
                outputs["anomaly_maps"], stats.pixel_mean, stats.pixel_std, center_at=stats.image_mean
            )

    def set_threshold(self, anomaly_module: AnomalyModule) -> None:
        """Sets threshold to 0.5

        Args:
            anomaly_module (AnomalyModule): Anomaly module.
        """
        for metric in (anomaly_module.image_metrics, anomaly_module.pixel_metrics):
            if metric is not None:
                metric.set_threshold(0.5)

    def _normalize_batch(self, outputs: STEP_OUTPUT, anomaly_module: AnomalyModule):
        """Normalize the batch.

        Args:
            outputs (STEP_OUTPUT): Output of the batch.
            anomaly_module (AnomalyModule): Anomaly module.
        """
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
        if self.normalization_method == NormalizationMethod.MIN_MAX:
            self._normalize_batch(outputs, anomaly_module)
        elif self.normalization_method == NormalizationMethod.CDF:
            self._standardize_batch(outputs)
            self._normalize_batch(outputs, anomaly_module)
