"""Density estimation module for AI-VAD model implementation."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from sklearn.mixture import GaussianMixture
from torch import Tensor, nn

from anomalib.models.ai_vad.features import FeatureType
from anomalib.utils.metrics.min_max import MinMax


class BaseDensityEstimator(nn.Module, ABC):
    """Base density estimator."""

    @abstractmethod
    def update(self, features: dict[FeatureType, Tensor] | Tensor, group: Any = None):
        """Update the density model with a new set of features."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, features: dict[FeatureType, Tensor] | Tensor):
        """Predict the density of a set of features."""
        raise NotImplementedError

    @abstractmethod
    def fit(self):
        """Compose model using collected features."""
        raise NotImplementedError

    def forward(self, features: dict[FeatureType, Tensor] | Tensor):
        """Update or predict depending on training status."""
        if self.training:
            self.update(features)
            return None
        return self.predict(features)


class CombinedDensityEstimator(BaseDensityEstimator):
    """Density estimator for AI-VAD.

    Combines density estimators for the different feature types included in the model.

    Args:
        use_velocity_features (bool): Flag indicating if velocity features should be used.
        use_pose_features (bool): Flag indicating if pose features should be used.
        use_deep_features (bool): Flag indicating if deep features should be used.
        n_components_velocity (int): Number of components used by GMM density estimation for velocity features.
        n_neighbors_pose (int): Number of neighbors used in KNN density estimation for pose features.
        n_neighbors_deep (int): Number of neighbors used in KNN density estimation for deep features.
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
        assert any((use_pose_features, use_deep_features, use_velocity_features))

    def update(self, features: dict[FeatureType, Tensor], group: Any = None) -> None:
        """Update the density estimators for the different feature types.

        Args:
            features (dict[FeatureType, Tensor]): Dictionary containing extracted features for a single frame.
            group (str): Identifier of the video from which the frame was sampled. Used for grouped density estimation.
        """
        if self.use_velocity_features:
            self.velocity_estimator.update(features[FeatureType.VELOCITY])
        if self.use_deep_features:
            self.appearance_estimator.update(features[FeatureType.DEEP], group=group)
        if self.use_pose_features:
            self.pose_estimator.update(features[FeatureType.POSE], group=group)

    def fit(self):
        """Fit the density estimation models on the collected features."""
        if self.use_velocity_features:
            self.velocity_estimator.fit()
        if self.use_deep_features:
            self.appearance_estimator.fit()
        if self.use_pose_features:
            self.pose_estimator.fit()

    def predict(self, features: dict[FeatureType, Tensor]) -> tuple[Tensor, Tensor]:
        """Predict the region- and image-level anomaly scores for an image based on a set of features.

        Args:
            features (dict[Tensor]): Dictionary containing extracted features for a single frame.

        Returns:
            Tensor: Region-level anomaly scores for all regions withing the frame.
            Tensor: Frame-level anomaly score for the frame.
        """
        n_regions = list(features.values())[0].shape[0]
        device = list(features.values())[0].device
        region_scores = torch.zeros(n_regions).to(device)
        image_score = 0
        if self.use_velocity_features:
            velocity_scores = self.velocity_estimator.predict(features[FeatureType.VELOCITY])
            region_scores += velocity_scores
            image_score += velocity_scores.max()
        if self.use_deep_features:
            deep_scores = self.appearance_estimator.predict(features[FeatureType.DEEP])
            region_scores += deep_scores
            image_score += deep_scores.max()
        if self.use_pose_features:
            pose_scores = self.pose_estimator.predict(features[FeatureType.POSE])
            region_scores += pose_scores
            image_score += pose_scores.max()
        return region_scores, image_score


class GroupedKNNEstimator(BaseDensityEstimator):
    """Grouped KNN density estimator.

    Keeps track of the group (e.g. video id) from which the features were sampled for normalization purposes.

    Args:
        n_neighbors (int): Number of neighbors used in KNN search.
    """

    def __init__(self, n_neighbors: int) -> None:
        super().__init__()

        self.n_neighbors = n_neighbors
        self.memory_bank: dict[Any, list[Tensor] | Tensor] = {}
        self.normalization_statistics = MinMax()

    def update(self, features: Tensor, group: Any = None) -> None:
        """Update the internal feature bank while keeping track of the group.

        Args:
            features (Tensor): Feature vectors extracted from a video frame.
            group (Any): Identifier of the group (video) from which the frame was sampled.
        """
        group = group or "default"

        if group in self.memory_bank:
            self.memory_bank[group].append(features)
        else:
            self.memory_bank[group] = [features]

    def fit(self) -> None:
        """Fit the KNN model by stacking the feature vectors and computing the normalization statistics."""
        self.memory_bank = {key: torch.vstack(value) for key, value in self.memory_bank.items()}
        self._compute_normalization_statistics()

    def predict(self, features: Tensor, group: Any = None, n_neighbors: int = 1, normalize: bool = True) -> Tensor:
        """Predict the (normalized) density for a set of features.

        Args:
            features (Tensor): Input features that will be compared to the density model.
            group (Any, optional): Group (video id) from which the features originate. If passed, all features of the
                same group in the memory bank will be excluded from the density estimation.
            n_neighbors (int): Number of neighbors used in the KNN search.
            normalize (bool): Flag indicating if the density should be normalized to min-max stats of the feature bank.
        Returns:
            Tensor: Mean (normalized) distances of input feature vectors to k nearest neighbors in feature bank.
        """
        n_neighbors = n_neighbors or self.n_neighbors

        if group:
            mem_bank = self.memory_bank.copy()
            mem_bank.pop(group)
        else:
            mem_bank = self.memory_bank

        mem_bank_tensor = torch.vstack(list(mem_bank.values()))

        distances = self._nearest_neighbors(mem_bank_tensor, features, n_neighbors=n_neighbors)

        if normalize:
            distances = self._normalize(distances)

        return distances.mean(axis=1)

    @staticmethod
    def _nearest_neighbors(feature_bank: Tensor, features: Tensor, n_neighbors: int = 1):
        """Perform the KNN search.

        Args:
            feature_bank (Tensor): Feature bank used for KNN search.
            features (Ternsor): Input features.
            n_neighbors (int): Number of neighbors used in KNN search.
        Returns:
            Tensor: Distances between the input features and their K nearest neighbors in the feature bank.
        """
        distances = torch.cdist(features, feature_bank, p=2.0)  # euclidean norm
        if n_neighbors == 1:
            # when n_neighbors is 1, speed up computation by using min instead of topk
            distances, _ = distances.min(1)
            return distances.unsqueeze(1)
        distances, _ = distances.topk(k=n_neighbors, largest=False, dim=1)
        return distances

    def _compute_normalization_statistics(self):
        """Compute min-max normalization statistics while taking the group into account."""
        for group, features in self.memory_bank.items():
            distances = self.predict(features, group, normalize=False)
            self.normalization_statistics.update(distances)

        self.normalization_statistics.compute()

    def _normalize(self, distances: Tensor):
        """Normalize distance predictions.

        Args:
            distances (Tensor): Distance tensor produced by KNN search.
        Returns:
            Tensor: Normalized distances.
        """
        return (distances - self.normalization_statistics.min) / (
            self.normalization_statistics.max - self.normalization_statistics.min
        )


class GMMEstimator(BaseDensityEstimator):
    """Density estimation based on Gaussian Mixture Model.

    Args:
        n_components (int): Number of components used in the GMM.
    """

    def __init__(self, n_components: int = 2) -> None:
        super().__init__()

        # TODO: replace with custom pytorch implementation of GMM (CVS-109432)
        self.gmm = GaussianMixture(n_components=n_components, random_state=0)
        self.memory_bank: list[Tensor] | Tensor = []

        self.normalization_statistics = MinMax()

    def update(self, features: Tensor, group: Any = None):
        """Update the feature bank."""
        del group
        self.memory_bank.append(features)

    def fit(self):
        """Fit the GMM and compute normalization statistics."""
        self.memory_bank = torch.vstack(self.memory_bank)
        self.gmm.fit(self.memory_bank.cpu())
        self._compute_normalization_statistics()

    def predict(self, features: Tensor, normalize: bool = True) -> Tensor:
        """Predict the density of a set of feature vectors.

        Args:
            features (Tensor): Input feature vectors.
            normalize (bool): Flag indicating if the density should be normalized to min-max stats of the feature bank.
        Returns:
            Tensor: Density scores of the input feature vectors.
        """
        density = -self.gmm.score_samples(features.cpu())
        density = Tensor(density).to(self.normalization_statistics.device)
        if normalize:
            density = self._normalize(density)
        return density

    def _compute_normalization_statistics(self):
        """Compute min-max normalization statistics over the feature bank."""
        training_scores = self.predict(self.memory_bank, normalize=False)
        self.normalization_statistics.update(training_scores)
        self.normalization_statistics.compute()

    def _normalize(self, density: Tensor):
        """Normalize distance predictions.

        Args:
            distances (Tensor): Distance tensor produced by KNN search.
        Returns:
            Tensor: Normalized distances.
        """
        return (density - self.normalization_statistics.min) / (
            self.normalization_statistics.max - self.normalization_statistics.min
        )
