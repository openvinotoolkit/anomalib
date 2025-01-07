"""Anomaly Map Generator for the PaDiM model implementation.

This module generates anomaly heatmaps for the PaDiM model by computing Mahalanobis
distances between test patch embeddings and reference distributions.

The anomaly map generation process involves:
1. Computing Mahalanobis distances between embeddings and reference statistics
2. Upsampling the distance map to match input image size
3. Applying Gaussian smoothing to obtain the final anomaly map

Example:
    >>> from anomalib.models.image.padim.anomaly_map import AnomalyMapGenerator
    >>> generator = AnomalyMapGenerator(sigma=4)
    >>> embedding = torch.randn(32, 1024, 28, 28)
    >>> mean = torch.randn(1024, 784)  # 784 = 28*28
    >>> inv_covariance = torch.randn(784, 1024, 1024)
    >>> anomaly_map = generator(
    ...     embedding=embedding,
    ...     mean=mean,
    ...     inv_covariance=inv_covariance,
    ...     image_size=(224, 224)
    ... )

See Also:
    - :class:`anomalib.models.image.padim.lightning_model.Padim`:
        Lightning implementation of the PaDiM model
    - :class:`anomalib.models.components.GaussianBlur2d`:
        Gaussian blur module used for smoothing anomaly maps
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.models.components import GaussianBlur2d


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap.

    This class implements anomaly map generation for the PaDiM model by computing
    Mahalanobis distances and applying post-processing steps.

    Args:
        sigma (int, optional): Standard deviation for Gaussian smoothing kernel.
            Higher values produce smoother anomaly maps. Defaults to ``4``.

    Example:
        >>> generator = AnomalyMapGenerator(sigma=4)
        >>> embedding = torch.randn(32, 1024, 28, 28)
        >>> mean = torch.randn(1024, 784)
        >>> inv_covariance = torch.randn(784, 1024, 1024)
        >>> anomaly_map = generator.compute_anomaly_map(
        ...     embedding=embedding,
        ...     mean=mean,
        ...     inv_covariance=inv_covariance,
        ...     image_size=(224, 224)
        ... )
    """

    def __init__(self, sigma: int = 4) -> None:
        super().__init__()
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.blur = GaussianBlur2d(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma), channels=1)

    @staticmethod
    def compute_distance(embedding: torch.Tensor, stats: list[torch.Tensor]) -> torch.Tensor:
        """Compute anomaly score for each patch position using Mahalanobis distance.

        Implements Equation (2) from Section III-C of the PaDiM paper to compute
        the distance between patch embeddings and their reference distributions.

        Args:
            embedding (torch.Tensor): Feature embeddings from the CNN backbone,
                shape ``(batch_size, n_features, height, width)``
            stats (list[torch.Tensor]): List containing mean and inverse covariance
                tensors for the multivariate Gaussian distributions

        Returns:
            torch.Tensor: Anomaly scores computed via Mahalanobis distance,
                shape ``(batch_size, 1, height, width)``
        """
        batch, channel, height, width = embedding.shape
        embedding = embedding.reshape(batch, channel, height * width)

        # calculate mahalanobis distances
        mean, inv_covariance = stats
        delta = (embedding - mean).permute(2, 0, 1)

        distances = (torch.matmul(delta, inv_covariance) * delta).sum(2).permute(1, 0)
        distances = distances.reshape(batch, 1, height, width)
        return distances.clamp(0).sqrt()

    @staticmethod
    def up_sample(distance: torch.Tensor, image_size: tuple[int, int] | torch.Size) -> torch.Tensor:
        """Up sample anomaly score to match the input image size.

        Args:
            distance (torch.Tensor): Anomaly scores, shape
                ``(batch_size, 1, height, width)``
            image_size (tuple[int, int] | torch.Size): Target size for upsampling,
                usually the original input image size

        Returns:
            torch.Tensor: Upsampled anomaly scores matching the input image size
        """
        return F.interpolate(
            distance,
            size=image_size,
            mode="bilinear",
            align_corners=False,
        )

    def smooth_anomaly_map(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian smoothing to the anomaly map.

        Args:
            anomaly_map (torch.Tensor): Raw anomaly scores,
                shape ``(batch_size, 1, height, width)``

        Returns:
            torch.Tensor: Smoothed anomaly scores with reduced noise
        """
        return self.blur(anomaly_map)

    def compute_anomaly_map(
        self,
        embedding: torch.Tensor,
        mean: torch.Tensor,
        inv_covariance: torch.Tensor,
        image_size: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        """Compute anomaly map from feature embeddings and distribution parameters.

        This method combines distance computation, upsampling, and smoothing to
        generate the final anomaly map.

        Args:
            embedding (torch.Tensor): Feature embeddings from the CNN backbone
            mean (torch.Tensor): Mean of the multivariate Gaussian distribution
            inv_covariance (torch.Tensor): Inverse covariance matrix
            image_size (tuple[int, int] | torch.Size | None, optional): Target
                size for upsampling. If ``None``, no upsampling is performed.

        Returns:
            torch.Tensor: Final anomaly map after all processing steps
        """
        score_map = self.compute_distance(
            embedding=embedding,
            stats=[mean.to(embedding.device), inv_covariance.to(embedding.device)],
        )
        if image_size:
            score_map = self.up_sample(score_map, image_size)
        return self.smooth_anomaly_map(score_map)

    def forward(self, **kwargs) -> torch.Tensor:
        """Generate anomaly map from the provided embeddings and statistics.

        Expects ``embedding``, ``mean`` and ``inv_covariance`` keywords to be
        passed explicitly.

        Example:
            >>> generator = AnomalyMapGenerator(sigma=4)
            >>> anomaly_map = generator(
            ...     embedding=embedding,
            ...     mean=mean,
            ...     inv_covariance=inv_covariance,
            ...     image_size=(224, 224)
            ... )

        Args:
            **kwargs: Keyword arguments containing ``embedding``, ``mean``,
                ``inv_covariance`` and optionally ``image_size``

        Raises:
            ValueError: If required keys are not found in ``kwargs``

        Returns:
            torch.Tensor: Generated anomaly map
        """
        if not ("embedding" in kwargs and "mean" in kwargs and "inv_covariance" in kwargs):
            msg = f"Expected keys `embedding`, `mean` and `covariance`. Found {kwargs.keys()}"
            raise ValueError(msg)

        embedding: torch.Tensor = kwargs["embedding"]
        mean: torch.Tensor = kwargs["mean"]
        inv_covariance: torch.Tensor = kwargs["inv_covariance"]
        image_size: tuple[int, int] | torch.Size = kwargs.get("image_size", None)

        return self.compute_anomaly_map(embedding, mean, inv_covariance, image_size=image_size)
