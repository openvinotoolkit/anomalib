"""Density estimation module for AI-VAD model implementation.

This module implements the density estimation stage of the AI-VAD model. It provides
density estimators for modeling the distribution of extracted features from normal
video samples.

The module provides the following components:
    - :class:`BaseDensityEstimator`: Abstract base class for density estimators
    - :class:`CombinedDensityEstimator`: Main density estimator that combines
      multiple feature-specific estimators

Example:
    >>> import torch
    >>> from anomalib.models.video.ai_vad.density import CombinedDensityEstimator
    >>> from anomalib.models.video.ai_vad.features import FeatureType
    >>> estimator = CombinedDensityEstimator()
    >>> features = {
    ...     FeatureType.VELOCITY: torch.randn(32, 8),
    ...     FeatureType.POSE: torch.randn(32, 34),
    ...     FeatureType.DEEP: torch.randn(32, 512)
    ... }
    >>> scores = estimator(features)  # Returns anomaly scores during inference

The density estimators are used to model the distribution of normal behavior and
detect anomalies as samples with low likelihood under the learned distributions.
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from anomalib.models.components.base import DynamicBufferMixin
from anomalib.models.components.cluster.gmm import GaussianMixture

from .features import FeatureType


class BaseDensityEstimator(nn.Module, ABC):
    """Abstract base class for density estimators.

    This class defines the interface for density estimators used in the AI-VAD model.
    Subclasses must implement methods for updating the density model with new features,
    predicting densities for test samples, and fitting the model.

    Example:
        >>> import torch
        >>> from anomalib.models.video.ai_vad.density import BaseDensityEstimator
        >>> class MyEstimator(BaseDensityEstimator):
        ...     def update(self, features, group=None):
        ...         pass
        ...     def predict(self, features):
        ...         return torch.rand(features.shape[0])
        ...     def fit(self):
        ...         pass
        >>> estimator = MyEstimator()
        >>> features = torch.randn(32, 8)
        >>> scores = estimator(features)  # Forward pass returns predictions
    """

    @abstractmethod
    def update(self, features: dict[FeatureType, torch.Tensor] | torch.Tensor, group: str | None = None) -> None:
        """Update the density model with a new set of features.

        Args:
            features (dict[FeatureType, torch.Tensor] | torch.Tensor): Input features
                to update the model. Can be either a dictionary mapping feature types
                to tensors, or a single tensor.
            group (str | None, optional): Optional group identifier for grouped
                density estimation. Defaults to ``None``.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        features: dict[FeatureType, torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Predict the density of a set of features.

        Args:
            features (dict[FeatureType, torch.Tensor] | torch.Tensor): Input features
                to compute density for. Can be either a dictionary mapping feature
                types to tensors, or a single tensor.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Predicted density
                scores. May return either a single tensor of scores or a tuple of
                tensors for more complex estimators.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self) -> None:
        """Compose model using collected features.

        This method should be called after updating the model with features to fit
        the density estimator to the collected data.
        """
        raise NotImplementedError

    def forward(
        self,
        features: dict[FeatureType, torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        """Forward pass that either updates or predicts based on training status.

        Args:
            features (dict[FeatureType, torch.Tensor] | torch.Tensor): Input
                features. Can be either a dictionary mapping feature types to
                tensors, or a single tensor.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None: During
                training, returns ``None`` after updating. During inference,
                returns density predictions.
        """
        if self.training:
            self.update(features)
            return None
        return self.predict(features)


class CombinedDensityEstimator(BaseDensityEstimator):
    """Density estimator for AI-VAD.

    Combines density estimators for the different feature types included in the model.

    Args:
        use_pose_features (bool, optional): Flag indicating if pose features should be
            used. Defaults to ``True``.
        use_deep_features (bool, optional): Flag indicating if deep features should be
            used. Defaults to ``True``.
        use_velocity_features (bool, optional): Flag indicating if velocity features
            should be used. Defaults to ``False``.
        n_neighbors_pose (int, optional): Number of neighbors used in KNN density
            estimation for pose features. Defaults to ``1``.
        n_neighbors_deep (int, optional): Number of neighbors used in KNN density
            estimation for deep features. Defaults to ``1``.
        n_components_velocity (int, optional): Number of components used by GMM density
            estimation for velocity features. Defaults to ``5``.

    Raises:
        ValueError: If none of the feature types (velocity, pose, deep) are enabled.

    Example:
        >>> from anomalib.models.video.ai_vad.density import CombinedDensityEstimator
        >>> estimator = CombinedDensityEstimator(
        ...     use_pose_features=True,
        ...     use_deep_features=True,
        ...     use_velocity_features=True,
        ...     n_neighbors_pose=1,
        ...     n_neighbors_deep=1,
        ...     n_components_velocity=5
        ... )
        >>> # Update with features from training data
        >>> estimator.update(features, group="video_001")
        >>> # Fit the density estimators
        >>> estimator.fit()
        >>> # Get predictions for test data
        >>> region_scores, image_score = estimator.predict(features)
    """

    def __init__(
        self,
        use_pose_features: bool = True,
        use_deep_features: bool = True,
        use_velocity_features: bool = False,
        n_neighbors_pose: int = 1,
        n_neighbors_deep: int = 1,
        n_components_velocity: int = 5,
    ) -> None:
        super().__init__()

        self.use_pose_features = use_pose_features
        self.use_deep_features = use_deep_features
        self.use_velocity_features = use_velocity_features

        if self.use_velocity_features:
            self.velocity_estimator = GMMEstimator(n_components=n_components_velocity)
        if self.use_deep_features:
            self.appearance_estimator = GroupedKNNEstimator(n_neighbors_deep)
        if self.use_pose_features:
            self.pose_estimator = GroupedKNNEstimator(n_neighbors=n_neighbors_pose)
        if not any((use_pose_features, use_deep_features, use_velocity_features)):
            msg = "At least one feature stream must be enabled."
            raise ValueError(msg)

    def update(self, features: dict[FeatureType, torch.Tensor], group: str | None = None) -> None:
        """Update the density estimators for the different feature types.

        Args:
            features (dict[FeatureType, torch.Tensor]): Dictionary containing
                extracted features for a single frame. Keys are feature types and
                values are the corresponding feature tensors.
            group (str | None, optional): Identifier of the video from which the
                frame was sampled. Used for grouped density estimation. Defaults to
                ``None``.
        """
        if self.use_velocity_features:
            self.velocity_estimator.update(features[FeatureType.VELOCITY])
        if self.use_deep_features:
            self.appearance_estimator.update(features[FeatureType.DEEP], group=group)
        if self.use_pose_features:
            self.pose_estimator.update(features[FeatureType.POSE], group=group)

    def fit(self) -> None:
        """Fit the density estimation models on the collected features.

        This method should be called after updating with all training features to
        fit the density estimators to the collected data.
        """
        if self.use_velocity_features:
            self.velocity_estimator.fit()
        if self.use_deep_features:
            self.appearance_estimator.fit()
        if self.use_pose_features:
            self.pose_estimator.fit()

    def predict(self, features: dict[FeatureType, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict region and image-level anomaly scores.

        Computes anomaly scores for each region in the frame and an overall frame
        score based on the maximum region score.

        Args:
            features (dict[FeatureType, torch.Tensor]): Dictionary containing
                extracted features for a single frame. Keys are feature types and
                values are the corresponding feature tensors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Region-level anomaly scores for all regions within the frame
                - Frame-level anomaly score for the frame

        Example:
            >>> features = {
            ...     FeatureType.VELOCITY: velocity_features,
            ...     FeatureType.DEEP: deep_features,
            ...     FeatureType.POSE: pose_features
            ... }
            >>> region_scores, image_score = estimator.predict(features)
        """
        n_regions = next(iter(features.values())).shape[0]
        device = next(iter(features.values())).device
        region_scores = torch.zeros(n_regions).to(device)
        image_score = 0
        if self.use_velocity_features and features[FeatureType.VELOCITY].numel():
            velocity_scores = self.velocity_estimator.predict(features[FeatureType.VELOCITY])
            region_scores += velocity_scores
            image_score += velocity_scores.max()
        if self.use_deep_features and features[FeatureType.DEEP].numel():
            deep_scores = self.appearance_estimator.predict(features[FeatureType.DEEP])
            region_scores += deep_scores
            image_score += deep_scores.max()
        if self.use_pose_features and features[FeatureType.POSE].numel():
            pose_scores = self.pose_estimator.predict(features[FeatureType.POSE])
            region_scores += pose_scores
            image_score += pose_scores.max()
        return region_scores, image_score


class GroupedKNNEstimator(DynamicBufferMixin, BaseDensityEstimator):
    """Grouped KNN density estimator.

    Keeps track of the group (e.g. video id) from which the features were sampled for
    normalization purposes.

    Args:
        n_neighbors (int): Number of neighbors used in KNN search.

    Example:
        >>> from anomalib.models.video.ai_vad.density import GroupedKNNEstimator
        >>> import torch
        >>> estimator = GroupedKNNEstimator(n_neighbors=5)
        >>> features = torch.randn(32, 512)  # (N, D)
        >>> estimator.update(features, group="video_1")
        >>> estimator.fit()
        >>> scores = estimator.predict(features)
        >>> scores.shape
        torch.Size([32])
    """

    def __init__(self, n_neighbors: int) -> None:
        """Initialize the grouped KNN density estimator.

        Args:
            n_neighbors (int): Number of neighbors used in KNN search.
        """
        super().__init__()

        self.n_neighbors = n_neighbors
        self.feature_collection: dict[str, list[torch.Tensor]] = {}
        self.group_index: dict[str, int] = {}

        self.register_buffer("memory_bank", Tensor())
        self.register_buffer("min", torch.tensor(torch.inf))
        self.register_buffer("max", torch.tensor(-torch.inf))

        self.memory_bank: torch.Tensor
        self.min: torch.Tensor
        self.max: torch.Tensor

    def update(self, features: torch.Tensor, group: str | None = None) -> None:
        """Update the internal feature bank while keeping track of the group.

        Args:
            features (torch.Tensor): Feature vectors extracted from a video frame of
                shape ``(N, D)``.
            group (str | None, optional): Identifier of the group (video) from which
                the frame was sampled. Defaults to ``None``.

        Example:
            >>> estimator = GroupedKNNEstimator(n_neighbors=5)
            >>> features = torch.randn(32, 512)  # (N, D)
            >>> estimator.update(features, group="video_1")
        """
        group = group or "default"

        if group in self.feature_collection:
            self.feature_collection[group].append(features)
        else:
            self.feature_collection[group] = [features]

    def fit(self) -> None:
        """Fit the KNN model by stacking features and computing normalization stats.

        Stacks the collected feature vectors group-wise and computes the normalization
        statistics. After fitting, the feature collection is deleted to free up memory.

        Example:
            >>> estimator = GroupedKNNEstimator(n_neighbors=5)
            >>> features = torch.randn(32, 512)  # (N, D)
            >>> estimator.update(features, group="video_1")
            >>> estimator.fit()
        """
        # stack the collected features group-wise
        feature_collection = {key: torch.vstack(value) for key, value in self.feature_collection.items()}
        # assign memory bank, group index and group names
        self.memory_bank = torch.vstack(list(feature_collection.values()))
        self.group_index = torch.repeat_interleave(
            Tensor([features.shape[0] for features in feature_collection.values()]).int(),
        )
        self.group_names = list(feature_collection.keys())
        self._compute_normalization_statistics(feature_collection)
        # delete the feature collection to free up memory
        del self.feature_collection

    def predict(
        self,
        features: torch.Tensor,
        group: str | None = None,
        n_neighbors: int = 1,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Predict the (normalized) density for a set of features.

        Args:
            features (torch.Tensor): Input features of shape ``(N, D)`` that will be
                compared to the density model.
            group (str | None, optional): Group (video id) from which the features
                originate. If passed, all features of the same group in the memory
                bank will be excluded from the density estimation.
                Defaults to ``None``.
            n_neighbors (int, optional): Number of neighbors used in the KNN search.
                Defaults to ``1``.
            normalize (bool, optional): Flag indicating if the density should be
                normalized to min-max stats of the feature bank.
                Defaults to ``True``.

        Returns:
            torch.Tensor: Mean (normalized) distances of input feature vectors to k
                nearest neighbors in feature bank.

        Example:
            >>> estimator = GroupedKNNEstimator(n_neighbors=5)
            >>> features = torch.randn(32, 512)  # (N, D)
            >>> estimator.update(features, group="video_1")
            >>> estimator.fit()
            >>> scores = estimator.predict(features, group="video_1")
            >>> scores.shape
            torch.Size([32])
        """
        n_neighbors = n_neighbors or self.n_neighbors

        if group:
            group_idx = self.group_names.index(group)
            mem_bank = self.memory_bank[self.group_index != group_idx]
        else:
            mem_bank = self.memory_bank

        distances = self._nearest_neighbors(mem_bank, features, n_neighbors=n_neighbors)

        if normalize:
            distances = self._normalize(distances)

        return distances.mean(axis=1)

    @staticmethod
    def _nearest_neighbors(feature_bank: torch.Tensor, features: torch.Tensor, n_neighbors: int = 1) -> torch.Tensor:
        """Perform the KNN search.

        Args:
            feature_bank (torch.Tensor): Feature bank of shape ``(M, D)`` used for
                KNN search.
            features (torch.Tensor): Input features of shape ``(N, D)``.
            n_neighbors (int, optional): Number of neighbors used in KNN search.
                Defaults to ``1``.

        Returns:
            torch.Tensor: Distances between the input features and their K nearest
                neighbors in the feature bank.
        """
        distances = torch.cdist(features, feature_bank, p=2.0)  # euclidean norm
        if n_neighbors == 1:
            # when n_neighbors is 1, speed up computation by using min instead of topk
            distances, _ = distances.min(1)
            return distances.unsqueeze(1)
        distances, _ = distances.topk(k=n_neighbors, largest=False, dim=1)
        return distances

    def _compute_normalization_statistics(self, grouped_features: dict[str, Tensor]) -> None:
        """Compute min-max normalization statistics while taking the group into account.

        Args:
            grouped_features (dict[str, Tensor]): Dictionary mapping group names to
                feature tensors.
        """
        for group, features in grouped_features.items():
            distances = self.predict(features, group, normalize=False)
            self.min = torch.min(self.min, torch.min(distances))
            self.max = torch.max(self.min, torch.max(distances))

    def _normalize(self, distances: torch.Tensor) -> torch.Tensor:
        """Normalize distance predictions.

        Args:
            distances (torch.Tensor): Distance tensor produced by KNN search.

        Returns:
            torch.Tensor: Normalized distances.
        """
        return (distances - self.min) / (self.max - self.min)


class GMMEstimator(BaseDensityEstimator):
    """Density estimation based on Gaussian Mixture Model.

    Fits a GMM to the training features and uses the negative log-likelihood as an
    anomaly score during inference.

    Args:
        n_components (int, optional): Number of Gaussian components used in the GMM.
            Defaults to ``2``.

    Example:
        >>> import torch
        >>> from anomalib.models.video.ai_vad.density import GMMEstimator
        >>> estimator = GMMEstimator(n_components=2)
        >>> features = torch.randn(32, 8)  # (N, D)
        >>> estimator.update(features)
        >>> estimator.fit()
        >>> scores = estimator.predict(features)
        >>> scores.shape
        torch.Size([32])
    """

    def __init__(self, n_components: int = 2) -> None:
        super().__init__()

        self.gmm = GaussianMixture(n_components=n_components)
        self.memory_bank: list[torch.Tensor] | torch.Tensor = []

        self.register_buffer("min", torch.tensor(torch.inf))
        self.register_buffer("max", torch.tensor(-torch.inf))

        self.min: torch.Tensor
        self.max: torch.Tensor

    def update(self, features: torch.Tensor, group: str | None = None) -> None:
        """Update the feature bank with new features.

        Args:
            features (torch.Tensor): Feature vectors of shape ``(N, D)`` to add to
                the memory bank.
            group (str | None, optional): Unused group parameter included for
                interface compatibility. Defaults to ``None``.
        """
        del group
        if isinstance(self.memory_bank, list):
            self.memory_bank.append(features)

    def fit(self) -> None:
        """Fit the GMM and compute normalization statistics.

        Concatenates all features in the memory bank, fits the GMM to the combined
        features, and computes min-max normalization statistics over the training
        scores.
        """
        self.memory_bank = torch.vstack(self.memory_bank)
        self.gmm.fit(self.memory_bank)
        self._compute_normalization_statistics()

    def predict(self, features: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Predict anomaly scores for input features.

        Computes the negative log-likelihood of each feature vector under the
        fitted GMM. Lower likelihood (higher score) indicates more anomalous
        samples.

        Args:
            features (torch.Tensor): Input feature vectors of shape ``(N, D)``.
            normalize (bool, optional): Whether to normalize scores using min-max
                statistics from training. Defaults to ``True``.

        Returns:
            torch.Tensor: Anomaly scores of shape ``(N,)``. Higher values indicate
                more anomalous samples.
        """
        density = -self.gmm.score_samples(features)
        if normalize:
            density = self._normalize(density)
        return density

    def _compute_normalization_statistics(self) -> None:
        """Compute min-max normalization statistics over the feature bank.

        Computes anomaly scores for all training features and updates the min-max
        statistics used for score normalization during inference.
        """
        training_scores = self.predict(self.memory_bank, normalize=False)
        self.min = torch.min(self.min, torch.min(training_scores))
        self.max = torch.max(self.min, torch.max(training_scores))

    def _normalize(self, density: torch.Tensor) -> torch.Tensor:
        """Normalize anomaly scores using min-max statistics.

        Args:
            density (torch.Tensor): Raw anomaly scores of shape ``(N,)``.

        Returns:
            torch.Tensor: Normalized anomaly scores of shape ``(N,)``.
        """
        return (density - self.min) / (self.max - self.min)
