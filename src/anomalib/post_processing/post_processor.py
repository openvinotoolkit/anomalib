# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Post-processing module for anomaly detection results.

This module provides post-processing functionality for one-class anomaly detection
outputs through the :class:`PostProcessor` class.

The post-processor handles:
    - Normalizing image and pixel-level anomaly scores
    - Computing adaptive thresholds for anomaly classification
    - Applying sensitivity adjustments to thresholds
    - Formatting results for downstream use

Example:
    >>> from anomalib.post_processing import PostProcessor
    >>> post_processor = PostProcessor(image_sensitivity=0.5)
    >>> predictions = post_processor(anomaly_maps=anomaly_maps)
"""

from typing import Any

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from torch import nn

from anomalib.data import Batch, InferenceBatch
from anomalib.metrics import F1AdaptiveThreshold, MinMax


class PostProcessor(nn.Module, Callback):
    """Post-processor for anomaly detection.

    This class handles post-processing of anomaly detection results by:
        - Normalizing image and pixel-level anomaly scores
        - Computing adaptive thresholds for anomaly classification
        - Applying sensitivity adjustments to thresholds
        - Formatting results for downstream use

    Args:
        enable_normalization (bool, optional): Enable normalization of anomaly scores.
            Defaults to True.
        enable_thresholding (bool, optional): Enable thresholding of anomaly scores.
            Defaults to True.
        enable_threshold_matching (bool, optional): Use image-level threshold for pixel-level predictions when
            pixel-level threshold is not available, and vice-versa.
            Defaults to True.
        image_sensitivity (float | None, optional): Sensitivity value for image-level
            predictions. Higher values make the model more sensitive to anomalies.
            Defaults to None.
        pixel_sensitivity (float | None, optional): Sensitivity value for pixel-level
            predictions. Higher values make the model more sensitive to anomalies.
            Defaults to None.
        **kwargs: Additional keyword arguments passed to parent class.

    Example:
        >>> from anomalib.post_processing import PostProcessor
        >>> post_processor = PostProcessor(image_sensitivity=0.5)
        >>> predictions = post_processor(anomaly_maps=anomaly_maps)
    """

    def __init__(
        self,
        enable_normalization: bool = True,
        enable_thresholding: bool = True,
        enable_threshold_matching: bool = True,
        image_sensitivity: float = 0.5,
        pixel_sensitivity: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.enable_thresholding = enable_thresholding
        self.enable_normalization = enable_normalization
        self.enable_threshold_matching = enable_threshold_matching

        # configure sensitivity values
        self.image_sensitivity = image_sensitivity
        self.pixel_sensitivity = pixel_sensitivity

        # initialize threshold and normalization metrics
        self._image_threshold_metric = F1AdaptiveThreshold(fields=["pred_score", "gt_label"], strict=False)
        self._pixel_threshold_metric = F1AdaptiveThreshold(fields=["anomaly_map", "gt_mask"], strict=False)
        self._image_min_max_metric = MinMax(fields=["pred_score"], strict=False)
        self._pixel_min_max_metric = MinMax(fields=["anomaly_map"], strict=False)

        # register buffers to persist threshold and normalization values
        self.register_buffer("_image_threshold", torch.tensor(float("nan")))
        self.register_buffer("_pixel_threshold", torch.tensor(float("nan")))
        self.register_buffer("image_min", torch.tensor(float("nan")))
        self.register_buffer("image_max", torch.tensor(float("nan")))
        self.register_buffer("pixel_min", torch.tensor(float("nan")))
        self.register_buffer("pixel_max", torch.tensor(float("nan")))

        self._image_threshold: torch.Tensor
        self._pixel_threshold: torch.Tensor
        self.image_min: torch.Tensor
        self.image_max: torch.Tensor
        self.pixel_min: torch.Tensor
        self.pixel_max: torch.Tensor

    @property
    def _checkpoint_config_keys(self) -> tuple[str, ...]:
        """Attribute names persisted by ``checkpoint_config``/``load_checkpoint_config``.

        Override in a subclass that manages additional configuration (see
        :class:`~anomalib.post_processing.MEBinPostProcessor`), typically by
        extending the inherited tuple rather than replacing it.

        Returns:
            tuple[str, ...]: Attribute names to persist.
        """
        return (
            "enable_normalization",
            "enable_thresholding",
            "enable_threshold_matching",
            "image_sensitivity",
            "pixel_sensitivity",
        )

    def checkpoint_config(self) -> dict[str, Any]:
        """Get plain-data configuration to persist in a checkpoint.

        Uses ``getattr`` rather than direct attribute access so that a subclass
        which does not call ``super().__init__()`` can still be checkpointed
        safely, instead of raising ``AttributeError`` during
        ``on_save_checkpoint``. Threshold and normalization statistics are not
        included here since they are already persisted via ``register_buffer``.

        Override this method (together with :meth:`load_checkpoint_config`) in a
        subclass that manages additional configuration, for example
        :class:`~anomalib.post_processing.MEBinPostProcessor`.

        Returns:
            dict[str, Any]: Plain-data configuration.
        """
        return {key: getattr(self, key, None) for key in self._checkpoint_config_keys}

    def load_checkpoint_config(self, config: dict[str, Any]) -> None:
        """Restore configuration previously returned by :meth:`checkpoint_config`.

        Args:
            config (dict[str, Any]): Plain-data configuration to restore.
        """
        for key in self._checkpoint_config_keys:
            if key in config:
                setattr(self, key, config[key])

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
        if self.enable_thresholding:
            # update threshold metrics
            self._image_threshold_metric.update(outputs)
            self._pixel_threshold_metric.update(outputs)
        if self.enable_normalization:
            # update normalization metrics
            self._image_min_max_metric.update(outputs)
            self._pixel_min_max_metric.update(outputs)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute final threshold and normalization values.

        Args:
            trainer (Trainer): PyTorch Lightning trainer instance.
            pl_module (LightningModule): PyTorch Lightning module instance.
        """
        del trainer, pl_module
        if self.enable_thresholding:
            # compute threshold values
            if self._image_threshold_metric.update_called:
                self._image_threshold.copy_(self._image_threshold_metric.compute())
                self._image_threshold_metric.reset()
            if self._pixel_threshold_metric.update_called:
                self._pixel_threshold.copy_(self._pixel_threshold_metric.compute())
                self._pixel_threshold_metric.reset()
        if self.enable_normalization:
            # compute normalization values
            if self._image_min_max_metric.update_called:
                image_min, image_max = self._image_min_max_metric.compute()
                self.image_min.copy_(image_min)
                self.image_max.copy_(image_max)
                self._image_min_max_metric.reset()
            if self._pixel_min_max_metric.update_called:
                pixel_min, pixel_max = self._pixel_min_max_metric.compute()
                self.pixel_min.copy_(pixel_min)
                self.pixel_max.copy_(pixel_max)
                self._pixel_min_max_metric.reset()

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
        pred_score = (
            predictions.pred_score
            if predictions.pred_score is not None
            else torch.amax(predictions.anomaly_map, dim=(-2, -1))
        )

        if self.enable_normalization:
            pred_score = self._normalize(pred_score, self.image_min, self.image_max, self.image_threshold)
            anomaly_map = self._normalize(predictions.anomaly_map, self.pixel_min, self.pixel_max, self.pixel_threshold)
        else:
            pred_score = predictions.pred_score
            anomaly_map = predictions.anomaly_map

        if self.enable_thresholding:
            pred_label = self._apply_threshold(pred_score, self.normalized_image_threshold)
            pred_mask = self._apply_threshold(anomaly_map, self.normalized_pixel_threshold)
        else:
            pred_label = None
            pred_mask = None

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
        if self.enable_normalization:
            self.normalize_batch(batch)
        # apply threshold
        if self.enable_thresholding:
            self.threshold_batch(batch)

    def threshold_batch(self, batch: Batch) -> None:
        """Apply thresholding to batch predictions.

        Args:
            batch (Batch): Batch containing model predictions.
        """
        batch.pred_label = (
            batch.pred_label
            if batch.pred_label is not None
            else self._apply_threshold(batch.pred_score, self.normalized_image_threshold)
        )
        batch.pred_mask = (
            batch.pred_mask
            if batch.pred_mask is not None
            else self._apply_threshold(batch.anomaly_map, self.normalized_pixel_threshold)
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
    def _apply_threshold(
        preds: torch.Tensor | None,
        threshold: torch.Tensor,
    ) -> torch.Tensor | None:
        """Apply thresholding to a single tensor.

        Args:
            preds (torch.Tensor | None): Predictions to threshold.
            threshold (float): Threshold value.

        Returns:
            torch.Tensor | None: Thresholded predictions or None if input is None.
        """
        if preds is None:
            return preds
        # Keep eager NaN behavior without introducing a data-dependent export guard.
        if not torch.compiler.is_compiling() and torch.isnan(threshold).item():
            return preds
        return preds > threshold

    @staticmethod
    def _normalize(
        preds: torch.Tensor | None,
        norm_min: torch.Tensor,
        norm_max: torch.Tensor,
        threshold: torch.Tensor,
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
            return preds

        threshold = torch.where(torch.isnan(threshold), (norm_max + norm_min) / 2, threshold)
        value_range = norm_max - norm_min
        zero_range = value_range == 0
        safe_range = torch.where(zero_range, torch.ones_like(value_range), value_range)
        normalized = (((preds - threshold) / safe_range) + 0.5).clamp(min=0, max=1)
        degenerate = torch.where(preds > threshold, 1.0, torch.where(preds < threshold, 0.0, 0.5))
        normalized = torch.where(zero_range, degenerate, normalized)
        has_range = ~torch.isnan(norm_min) & ~torch.isnan(norm_max)
        return torch.where(has_range, normalized, preds)

    @property
    def image_threshold(self) -> torch.Tensor:
        """Get the image-level threshold.

        Returns:
            float: Image-level threshold value.
        """
        fallback = (
            self._pixel_threshold
            if self.enable_threshold_matching
            else torch.full_like(self._image_threshold, float("nan"))
        )
        return torch.where(torch.isnan(self._image_threshold), fallback, self._image_threshold)

    @property
    def pixel_threshold(self) -> torch.Tensor:
        """Get the pixel-level threshold.

        If the pixel-level threshold is not set, the image-level threshold is used.

        Returns:
            float: Pixel-level threshold value.
        """
        fallback = (
            self._image_threshold
            if self.enable_threshold_matching
            else torch.full_like(self._pixel_threshold, float("nan"))
        )
        return torch.where(torch.isnan(self._pixel_threshold), fallback, self._pixel_threshold)

    @property
    def normalized_image_threshold(self) -> torch.Tensor:
        """Get the normalized image-level threshold.

        Returns:
            float: Normalized image-level threshold value, adjusted by sensitivity.
        """
        return torch.tensor(1.0) - self.image_sensitivity

    @property
    def normalized_pixel_threshold(self) -> torch.Tensor:
        """Get the normalized pixel-level threshold.

        Returns:
            float: Normalized pixel-level threshold value, adjusted by sensitivity.
        """
        return torch.tensor(1.0) - self.pixel_sensitivity
