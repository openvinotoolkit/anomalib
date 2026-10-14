# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MEBin post-processor for anomaly detection.

This module provides a post-processor that uses the MEBin (Main Element
Binarization) algorithm to adaptively determine per-image thresholds for anomaly
map binarization. Unlike the default :class:`PostProcessor` which uses a single
global F1-adaptive threshold, `MEBinPostProcessor` computes a per-image
threshold based on connected-component stability analysis.

Reference:
    "AnomalyNCD: Towards Novel Anomaly Class Discovery in
    Industrial Scenarios", CVPR 2025.
    https://arxiv.org/abs/2410.14379
    https://github.com/HUST-SLOW/AnomalyNCD



Example:
    >>> from anomalib.models import Padim
    >>> from anomalib.post_processing import MEBinPostProcessor
    >>> post_processor = MEBinPostProcessor()
    >>> model = Padim(post_processor=post_processor)
"""

import torch

from anomalib.data import Batch, InferenceBatch

from .mebin import mebin_binarize
from .post_processor import PostProcessor


class MEBinPostProcessor(PostProcessor):
    """Post-processor using MEBin adaptive binarization.

    MEBin determines per-image thresholds by sweeping thresholds across the
    anomaly map, counting connected components at each level, and selecting the
    threshold at the endpoint of the longest stable interval (a contiguous
    range where the component count stays constant).

    This post-processor inherits all normalization and metric-tracking
    functionality from :class:`PostProcessor`. The key difference is that
    `pred_mask` is computed using the MEBin per-image threshold instead of the
    global F1-adaptive threshold.

    .. note::
        MEBin is precision-oriented — it suppresses false positives at the cost
        of recall.  Pixel-level F1 might be lower than the default
        post-processor, but the resulting masks are better suited for downstream
        tasks like anomaly class discovery.

    Args:
        sample_rate: Step size for the threshold sweep in the normalised
            [0, 255] space. Smaller values give finer granularity but are
            slower. Defaults to `4`.
        min_interval_len: Minimum length (in sweep steps) of a stable interval
            to be considered valid. Defaults to `4`.
        erode: Whether to apply morphological erosion to the binarized map
            before counting connected components. Helps suppress noise.
            Defaults to `True`.
        kernel_size: Size of the square erosion kernel (only used when
            `erode=True`). Defaults to `6`.
        **kwargs: Additional keyword arguments passed to
            :class:`PostProcessor`.

    Example:
        >>> from anomalib.post_processing import MEBinPostProcessor
        >>> processor = MEBinPostProcessor(sample_rate=4, min_interval_len=4)
        >>> # Use with a model
        >>> from anomalib.models import Patchcore
        >>> model = Patchcore(post_processor=processor)
    """

    def __init__(
        self,
        sample_rate: int = 4,
        min_interval_len: int = 4,
        erode: bool = True,
        kernel_size: int = 6,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.min_interval_len = min_interval_len
        self.erode = erode
        self.kernel_size = kernel_size

    @property
    def _checkpoint_config_keys(self) -> tuple[str, ...]:
        """Attribute names persisted by ``checkpoint_config``/``load_checkpoint_config``.

        Extends the base :class:`PostProcessor` keys with MEBin's own
        configuration, so a checkpoint saved with a custom ``sample_rate``,
        ``min_interval_len``, ``erode``, or ``kernel_size`` restores them too.

        Returns:
            tuple[str, ...]: Attribute names to persist.
        """
        return (
            *super()._checkpoint_config_keys,
            "sample_rate",
            "min_interval_len",
            "erode",
            "kernel_size",
        )

    def forward(self, predictions: InferenceBatch) -> InferenceBatch:
        """Post-process model predictions using MEBin adaptive thresholding.

        When normalization statistics (``pixel_min`` / ``pixel_max``) are
        available (i.e. after a validation pass), the anomaly maps are first
        normalised using the standard :class:`PostProcessor` logic, and then
        MEBin is applied to compute per-image masks.

        When normalization statistics are not available (e.g. during
        standalone inference), MEBin is applied directly to the raw anomaly
        maps.

        Args:
            predictions: Batch of model predictions containing at least one of
                `pred_score` or `anomaly_map`.

        Returns:
            InferenceBatch: Post-processed predictions with:
                - pred_score -- normalised image-level anomaly score
                - anomaly_map -- normalised anomaly map
                - pred_label -- binary image-level label (from global
                  threshold)
                - pred_mask -- binary pixel-level mask (from MEBin)

        Raises:
            ValueError: If neither `pred_score` nor `anomaly_map` is
                provided.
        """
        if predictions.pred_score is None and predictions.anomaly_map is None:
            msg = "At least one of pred_score or anomaly_map must be provided."
            raise ValueError(msg)

        anomaly_map = predictions.anomaly_map
        pred_score = (
            predictions.pred_score if predictions.pred_score is not None else torch.amax(anomaly_map, dim=(-2, -1))
        )

        # --- Normalize using inherited logic (same as base PostProcessor) ---
        if self.enable_normalization:
            pred_score = self._normalize(pred_score, self.image_min, self.image_max, self.image_threshold)
            if anomaly_map is not None:
                anomaly_map = self._normalize(anomaly_map, self.pixel_min, self.pixel_max, self.pixel_threshold)

        # --- MEBin adaptive mask ---
        # MEBin works on the (possibly normalised) anomaly map.
        pred_mask: torch.Tensor | None = None
        if anomaly_map is not None and self.enable_thresholding:
            # Ensure (B, 1, H, W) shape for mebin_binarize.
            maps_4d = anomaly_map
            if maps_4d.dim() == 3:
                maps_4d = maps_4d.unsqueeze(1)
            pred_mask, _ = mebin_binarize(
                maps_4d,
                sample_rate=self.sample_rate,
                min_interval_len=self.min_interval_len,
                erode=self.erode,
                kernel_size=self.kernel_size,
            )
            # Squeeze back if anomaly_map was 3-D.
            if predictions.anomaly_map is not None and predictions.anomaly_map.dim() == 3:
                pred_mask = pred_mask.squeeze(1)
            pred_mask = pred_mask > 0.5

        # --- Image-level label uses the standard global threshold ---
        pred_label: torch.Tensor | None = None
        if self.enable_thresholding:
            pred_label = self._apply_threshold(pred_score, self.normalized_image_threshold)

        return InferenceBatch(
            pred_label=pred_label,
            pred_score=pred_score,
            pred_mask=pred_mask,
            anomaly_map=anomaly_map,
        )

    def post_process_batch(self, batch: Batch) -> None:
        """Post-process a batch during `on_test_batch_end` / `on_predict_batch_end`.

        Applies the inherited normalization (image + pixel level) and then
        computes the pixel-level mask using MEBin instead of the global
        threshold.

        Args:
            batch: Batch containing model predictions (modified in-place).
        """
        # 1. Normalize (same as base).
        if self.enable_normalization:
            self.normalize_batch(batch)

        # 2. Image-level threshold (same as base).
        if self.enable_thresholding:
            batch.pred_label = (
                batch.pred_label
                if batch.pred_label is not None
                else self._apply_threshold(batch.pred_score, self.normalized_image_threshold)
            )

        # 3. Pixel-level mask via MEBin.
        if self.enable_thresholding and batch.anomaly_map is not None and batch.pred_mask is None:
            maps_4d = batch.anomaly_map
            if maps_4d.dim() == 3:
                maps_4d = maps_4d.unsqueeze(1)
            pred_mask, _ = mebin_binarize(
                maps_4d,
                sample_rate=self.sample_rate,
                min_interval_len=self.min_interval_len,
                erode=self.erode,
                kernel_size=self.kernel_size,
            )
            if batch.anomaly_map.dim() == 3:
                pred_mask = pred_mask.squeeze(1)
            batch.pred_mask = pred_mask > 0.5
