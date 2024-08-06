"""Post-processing module for anomaly detection models."""

from abc import ABC, abstractmethod

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from torch import nn

from anomalib.dataclasses import Batch, InferenceBatch
from anomalib.metrics import F1AdaptiveThreshold, MinMax


class PostProcessor(nn.Module, Callback, ABC):
    """Base class for post-processor.

    The post-processor is a callback that is used to post-process the predictions of the model.
    """

    @abstractmethod
    def forward(self, batch: InferenceBatch) -> InferenceBatch:
        """Functional forward method for post-processing."""


class OneClassPostProcessor(PostProcessor):
    """Default post-processor for one-class anomaly detection."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._image_threshold = F1AdaptiveThreshold()
        self._pixel_threshold = F1AdaptiveThreshold()
        self._normalization_stats = MinMax()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Batch,
        *args,
        **kwargs,
    ) -> None:
        """Update the normalization and thresholding metrics using the batch output."""
        del trainer, pl_module, args, kwargs  # Unused arguments.
        self._image_threshold.update(outputs.pred_score, outputs.gt_label)
        self._pixel_threshold.update(outputs.anomaly_map, outputs.gt_mask)
        self._normalization_stats.update(outputs.anomaly_map)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute the final threshold and normalization values."""
        del trainer, pl_module
        self._image_threshold.compute()
        self._pixel_threshold.compute()
        self._normalization_stats.compute()

    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Batch, *args, **kwargs) -> None:
        """Apply the post-processing steps to the current batch of predictions."""
        del trainer, pl_module, args, kwargs
        self.post_process_batch(outputs)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Batch,
        *args,
        **kwargs,
    ) -> None:
        """Normalize the predicted scores and anomaly maps."""
        del trainer, pl_module, args, kwargs
        self.post_process_batch(outputs)

    def forward(self, predictions: InferenceBatch) -> InferenceBatch:
        """Funcional forward method for post-processing."""
        assert predictions.anomaly_map is not None, "Anomaly map is required for one-class post-processing."
        pred_score = predictions.pred_score or torch.amax(predictions.anomaly_map, dim=(-2, -1))
        pred_label = self._threshold(pred_score, self.image_threshold)
        pred_mask = self._threshold(predictions.anomaly_map, self.pixel_threshold)
        pred_score = self._normalize(pred_score, self.min, self.max, self.image_threshold)
        anomaly_map = self._normalize(predictions.anomaly_map, self.min, self.max, self.pixel_threshold)
        return InferenceBatch(
            pred_label=pred_label,
            pred_score=pred_score,
            pred_mask=pred_mask,
            anomaly_map=anomaly_map,
        )

    def post_process_batch(self, batch: Batch) -> None:
        """Normalize the predicted scores and anomaly maps."""
        # apply threshold
        self.threshold_batch(batch)
        # apply normalization
        self.normalize_batch(batch)

    def threshold_batch(self, batch: Batch) -> None:
        """Apply thresholding to the batch predictions."""
        pred_label = batch.pred_label or self._threshold(batch.pred_score, self.image_threshold)
        pred_mask = batch.pred_mask or self._threshold(batch.anomaly_map, self.pixel_threshold)
        batch.replace(pred_label=pred_label, pred_mask=pred_mask)

    def normalize_batch(self, batch: Batch) -> None:
        """Normalize the predicted scores and anomaly maps."""
        # normalize image-level predictions
        pred_score = self._normalize(batch.pred_score, self.min, self.max, self.image_threshold)
        # normalize pixel-level predictions
        anomaly_map = self._normalize(batch.anomaly_map, self.min, self.max, self.pixel_threshold)
        batch.replace(pred_score=pred_score, anomaly_map=anomaly_map)

    @staticmethod
    def _threshold(preds: torch.Tensor, threshold: float) -> torch.Tensor:
        """Apply thresholding to a single tensor."""
        return preds > threshold

    @staticmethod
    def _normalize(preds: torch.Tensor, norm_min: float, norm_max: float, threshold: float) -> torch.Tensor:
        """Normalize a tensor using the min, max, and threshold values."""
        preds = ((preds - threshold) / (norm_max - norm_min)) + 0.5
        preds = torch.minimum(preds, torch.tensor(1))
        return torch.maximum(preds, torch.tensor(0))

    @property
    def image_threshold(self) -> float:
        """Get the image-level threshold."""
        return self._image_threshold.value

    @property
    def pixel_threshold(self) -> float:
        """Get the pixel-level threshold."""
        return self._pixel_threshold.value

    @property
    def min(self) -> float:
        """Get the minimum value for normalization."""
        return self._normalization_stats.min

    @property
    def max(self) -> float:
        """Get the maximum value for normalization."""
        return self._normalization_stats.max
