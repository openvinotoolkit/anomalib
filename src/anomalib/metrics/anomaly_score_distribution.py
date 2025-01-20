"""Compute statistics of anomaly score distributions.

This module provides the ``AnomalyScoreDistribution`` class which computes mean
and standard deviation statistics of anomaly scores from normal training data.
Statistics are computed for both image-level and pixel-level scores.

The class tracks:
    - Image-level statistics: Mean and std of image anomaly scores
    - Pixel-level statistics: Mean and std of pixel anomaly maps

Example:
    >>> from anomalib.metrics import AnomalyScoreDistribution
    >>> import torch
    >>> # Create sample data
    >>> scores = torch.tensor([0.1, 0.2, 0.15])  # Image anomaly scores
    >>> maps = torch.tensor([[0.1, 0.2], [0.15, 0.25]])  # Pixel anomaly maps
    >>> # Initialize and compute stats
    >>> dist = AnomalyScoreDistribution()
    >>> dist.update(anomaly_scores=scores, anomaly_maps=maps)
    >>> image_mean, image_std, pixel_mean, pixel_std = dist.compute()

Note:
    The input scores and maps are log-transformed before computing statistics.
    Both image-level scores and pixel-level maps are optional inputs.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torchmetrics import Metric


class AnomalyScoreDistribution(Metric):
    """Compute distribution statistics of anomaly scores.

    This class tracks and computes the mean and standard deviation of anomaly
    scores from the normal samples in the training set. Statistics are computed
    for both image-level scores and pixel-level anomaly maps.

    The metric maintains internal state to accumulate scores and maps across
    batches before computing final statistics.

    Example:
        >>> dist = AnomalyScoreDistribution()
        >>> # Update with batch of scores
        >>> scores = torch.tensor([0.1, 0.2, 0.3])
        >>> dist.update(anomaly_scores=scores)
        >>> # Compute statistics
        >>> img_mean, img_std, pix_mean, pix_std = dist.compute()
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the metric states.

        Args:
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self.anomaly_maps: list[torch.Tensor] = []
        self.anomaly_scores: list[torch.Tensor] = []

        self.add_state("image_mean", torch.empty(0), persistent=True)
        self.add_state("image_std", torch.empty(0), persistent=True)
        self.add_state("pixel_mean", torch.empty(0), persistent=True)
        self.add_state("pixel_std", torch.empty(0), persistent=True)

        self.image_mean = torch.empty(0)
        self.image_std = torch.empty(0)
        self.pixel_mean = torch.empty(0)
        self.pixel_std = torch.empty(0)

    def update(
        self,
        *args,
        anomaly_scores: torch.Tensor | None = None,
        anomaly_maps: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """Update the internal state with new scores and maps.

        Args:
            *args: Unused positional arguments.
            anomaly_scores: Batch of image-level anomaly scores.
            anomaly_maps: Batch of pixel-level anomaly maps.
            **kwargs: Unused keyword arguments.
        """
        del args, kwargs  # These variables are not used.

        if anomaly_maps is not None:
            self.anomaly_maps.append(anomaly_maps)
        if anomaly_scores is not None:
            self.anomaly_scores.append(anomaly_scores)

    def compute(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute distribution statistics from accumulated scores and maps.

        Returns:
            tuple containing:
                - image_mean: Mean of log-transformed image anomaly scores
                - image_std: Standard deviation of log-transformed image scores
                - pixel_mean: Mean of log-transformed pixel anomaly maps
                - pixel_std: Standard deviation of log-transformed pixel maps
        """
        anomaly_scores = torch.hstack(self.anomaly_scores)
        anomaly_scores = torch.log(anomaly_scores)

        self.image_mean = anomaly_scores.mean()
        self.image_std = anomaly_scores.std()

        if self.anomaly_maps:
            anomaly_maps = torch.vstack(self.anomaly_maps)
            anomaly_maps = torch.log(anomaly_maps).cpu()

            self.pixel_mean = anomaly_maps.mean(dim=0).squeeze()
            self.pixel_std = anomaly_maps.std(dim=0).squeeze()

        return self.image_mean, self.image_std, self.pixel_mean, self.pixel_std
