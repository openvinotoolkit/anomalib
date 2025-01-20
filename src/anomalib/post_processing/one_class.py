"""Post-processing module for one-class anomaly detection results.

This module provides post-processing functionality for one-class anomaly detection
outputs through the :class:`OneClassPostProcessor` class.

The post-processor handles:
    - Normalizing image and pixel-level anomaly scores
    - Computing adaptive thresholds for anomaly classification
    - Applying sensitivity adjustments to thresholds
    - Formatting results for downstream use

Example:
    >>> from anomalib.post_processing import OneClassPostProcessor
    >>> post_processor = OneClassPostProcessor(image_sensitivity=0.5)
    >>> predictions = post_processor(anomaly_maps=anomaly_maps)
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from lightning import LightningModule, Trainer

from anomalib.data import Batch, InferenceBatch
from anomalib.metrics import F1AdaptiveThreshold, MinMax

from .base import PostProcessor


class OneClassPostProcessor(PostProcessor):
    """Post-processor for one-class anomaly detection.

    This class handles post-processing of anomaly detection results by:
        - Normalizing image and pixel-level anomaly scores
        - Computing adaptive thresholds for anomaly classification
        - Applying sensitivity adjustments to thresholds
        - Formatting results for downstream use

    Args:
        image_sensitivity (float | None, optional): Sensitivity value for image-level
            predictions. Higher values make the model more sensitive to anomalies.
            Defaults to None.
        pixel_sensitivity (float | None, optional): Sensitivity value for pixel-level
            predictions. Higher values make the model more sensitive to anomalies.
            Defaults to None.
        **kwargs: Additional keyword arguments passed to parent class.

    Example:
        >>> from anomalib.post_processing import OneClassPostProcessor
        >>> post_processor = OneClassPostProcessor(image_sensitivity=0.5)
        >>> predictions = post_processor(anomaly_maps=anomaly_maps)
    """

    def __init__(
        self,
        image_sensitivity: float | None = None,
        pixel_sensitivity: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # configure sensitivity values
        self.image_sensitivity = image_sensitivity
        self.pixel_sensitivity = pixel_sensitivity

        # initialize threshold and normalization metrics
        self._image_threshold = F1AdaptiveThreshold()
        self._pixel_threshold = F1AdaptiveThreshold()
        self._image_normalization_stats = MinMax()
        self._pixel_normalization_stats = MinMax()

        # register buffers to persist threshold and normalization values
        self.register_buffer("image_threshold", torch.tensor(0))
        self.register_buffer("pixel_threshold", torch.tensor(0))
        self.register_buffer("image_min", torch.tensor(0))
        self.register_buffer("image_max", torch.tensor(1))
        self.register_buffer("pixel_min", torch.tensor(0))
        self.register_buffer("pixel_max", torch.tensor(1))

        self.image_threshold: torch.Tensor
        self.pixel_threshold: torch.Tensor
        self.image_min: torch.Tensor
        self.image_max: torch.Tensor
        self.pixel_min: torch.Tensor
        self.pixel_max: torch.Tensor

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Batch,
        *args,
        **kwargs,
    ) -> None:
        """Update normalization and thresholding metrics using batch output.

        Args:
            trainer (Trainer): PyTorch Lightning trainer instance.
            pl_module (LightningModule): PyTorch Lightning module instance.
            outputs (Batch): Batch containing model predictions and ground truth.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        del trainer, pl_module, args, kwargs  # Unused arguments.
        if outputs.pred_score is not None:
            self._image_threshold.update(outputs.pred_score, outputs.gt_label)
        if outputs.anomaly_map is not None:
            self._pixel_threshold.update(outputs.anomaly_map, outputs.gt_mask)
        if outputs.pred_score is not None:
            self._image_normalization_stats.update(outputs.pred_score)
        if outputs.anomaly_map is not None:
            self._pixel_normalization_stats.update(outputs.anomaly_map)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute final threshold and normalization values.

        Args:
            trainer (Trainer): PyTorch Lightning trainer instance.
            pl_module (LightningModule): PyTorch Lightning module instance.
        """
        del trainer, pl_module
        if self._image_threshold.update_called:
            self.image_threshold = self._image_threshold.compute()
        if self._pixel_threshold.update_called:
            self.pixel_threshold = self._pixel_threshold.compute()
        if self._image_normalization_stats.update_called:
            self.image_min, self.image_max = self._image_normalization_stats.compute()
        if self._pixel_normalization_stats.update_called:
            self.pixel_min, self.pixel_max = self._pixel_normalization_stats.compute()

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Batch,
        *args,
        **kwargs,
    ) -> None:
        """Apply post-processing steps to current batch of predictions.

        Args:
            trainer (Trainer): PyTorch Lightning trainer instance.
            pl_module (LightningModule): PyTorch Lightning module instance.
            outputs (Batch): Batch containing model predictions.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
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
        """Normalize predicted scores and anomaly maps.

        Args:
            trainer (Trainer): PyTorch Lightning trainer instance.
            pl_module (LightningModule): PyTorch Lightning module instance.
            outputs (Batch): Batch containing model predictions.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        del trainer, pl_module, args, kwargs
        self.post_process_batch(outputs)

    def forward(self, predictions: InferenceBatch) -> InferenceBatch:
        """Post-process model predictions.

        Args:
            predictions (InferenceBatch): Batch containing model predictions.

        Returns:
            InferenceBatch: Post-processed batch with normalized scores and
                thresholded predictions.

        Raises:
            ValueError: If neither `pred_score` nor `anomaly_map` is provided.
        """
        if predictions.pred_score is None and predictions.anomaly_map is None:
            msg = "At least one of pred_score or anomaly_map must be provided."
            raise ValueError(msg)
        pred_score = predictions.pred_score or torch.amax(predictions.anomaly_map, dim=(-2, -1))
        pred_score = self._normalize(pred_score, self.image_min, self.image_max, self.image_threshold)
        anomaly_map = self._normalize(predictions.anomaly_map, self.pixel_min, self.pixel_max, self.pixel_threshold)
        pred_label = self._threshold(pred_score, self.normalized_image_threshold)
        pred_mask = self._threshold(anomaly_map, self.normalized_pixel_threshold)
        return InferenceBatch(
            pred_label=pred_label,
            pred_score=pred_score,
            pred_mask=pred_mask,
            anomaly_map=anomaly_map,
        )

    def post_process_batch(self, batch: Batch) -> None:
        """Post-process a batch of predictions.

        Applies normalization and thresholding to the batch predictions.

        Args:
            batch (Batch): Batch containing model predictions.
        """
        # apply normalization
        self.normalize_batch(batch)
        # apply threshold
        self.threshold_batch(batch)

    def threshold_batch(self, batch: Batch) -> None:
        """Apply thresholding to batch predictions.

        Args:
            batch (Batch): Batch containing model predictions.
        """
        batch.pred_label = (
            batch.pred_label
            if batch.pred_label is not None
            else self._threshold(batch.pred_score, self.normalized_image_threshold)
        )
        batch.pred_mask = (
            batch.pred_mask
            if batch.pred_mask is not None
            else self._threshold(batch.anomaly_map, self.normalized_pixel_threshold)
        )

    def normalize_batch(self, batch: Batch) -> None:
        """Normalize predicted scores and anomaly maps.

        Args:
            batch (Batch): Batch containing model predictions.
        """
        # normalize pixel-level predictions
        batch.anomaly_map = self._normalize(batch.anomaly_map, self.pixel_min, self.pixel_max, self.pixel_threshold)
        # normalize image-level predictions
        batch.pred_score = self._normalize(batch.pred_score, self.image_min, self.image_max, self.image_threshold)

    @staticmethod
    def _threshold(preds: torch.Tensor | None, threshold: float) -> torch.Tensor | None:
        """Apply thresholding to a single tensor.

        Args:
            preds (torch.Tensor | None): Predictions to threshold.
            threshold (float): Threshold value.

        Returns:
            torch.Tensor | None: Thresholded predictions or None if input is None.
        """
        if preds is None:
            return None
        return preds > threshold

    @staticmethod
    def _normalize(
        preds: torch.Tensor | None,
        norm_min: float,
        norm_max: float,
        threshold: float,
    ) -> torch.Tensor | None:
        """Normalize a tensor using min, max, and threshold values.

        Args:
            preds (torch.Tensor | None): Predictions to normalize.
            norm_min (float): Minimum value for normalization.
            norm_max (float): Maximum value for normalization.
            threshold (float): Threshold value.

        Returns:
            torch.Tensor | None: Normalized predictions or None if input is None.
        """
        if preds is None:
            return None
        preds = ((preds - threshold) / (norm_max - norm_min)) + 0.5
        return preds.clamp(min=0, max=1)

    @property
    def normalized_image_threshold(self) -> float:
        """Get the normalized image-level threshold.

        Returns:
            float: Normalized image-level threshold value, adjusted by sensitivity.
        """
        if self.image_sensitivity is not None:
            return 1 - self.image_sensitivity
        return 0.5

    @property
    def normalized_pixel_threshold(self) -> float:
        """Get the normalized pixel-level threshold.

        Returns:
            float: Normalized pixel-level threshold value, adjusted by sensitivity.
        """
        if self.pixel_sensitivity is not None:
            return 1 - self.pixel_sensitivity
        return 0.5
