
from __future__ import annotations

import torch
from torch import nn, Tensor
from typing import Any

from abc import ABC, abstractmethod
from anomalib.utils.metrics.min_max import MinMax
from anomalib.models.ai_vad.features import FeatureType
from sklearn.mixture import GaussianMixture


class BaseDensityEstimator(nn.Module, ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def update(self, features):
        raise NotImplementedError

    @abstractmethod
    def predict(self, features):
        raise NotImplementedError

    @abstractmethod
    def fit(self):
        """Compose models using collected features"""
        raise NotImplementedError

    def forward(self, features):
        if self.training:
            self.update(features)
        else:
            return self.predict(features)


class CombinedDensityEstimator(BaseDensityEstimator):

    def __init__(
            self,
            use_pose_features: bool = True,
            use_appearance_features: bool = True,
            use_velocity_features: bool = False,
            n_neighbors_pose: int = 1,
            n_neighbors_appearance: int = 1,
            n_components_velocity: int = 5
        ) -> None:
        super().__init__()

        self.use_pose_features = use_pose_features
        self.use_appearance_features = use_appearance_features
        self.use_velocity_features = use_velocity_features

        if self.use_velocity_features:
            self.velocity_estimator = GMMEstimator(n_components=n_components_velocity)
        if self.use_appearance_features:
            self.appearance_estimator = GroupedKNNEstimator(n_neighbors_appearance)
        if self.use_pose_features:
            self.pose_estimator = GroupedKNNEstimator(n_neighbors=n_neighbors_pose)
        assert any((use_pose_features, use_appearance_features, use_velocity_features))

    def update(self, features, video_id):
        if self.use_velocity_features:
            self.velocity_estimator.update(features[FeatureType.VELOCITY])
        if self.use_appearance_features:
            self.appearance_estimator.update(features[FeatureType.APPEARANCE], group=video_id)
        if self.use_pose_features:
            self.pose_estimator.update(features[FeatureType.POSE], group=video_id)

    def fit(self):
        if self.use_velocity_features:
            self.velocity_estimator.fit()
        if self.use_appearance_features:
            self.appearance_estimator.fit()
        if self.use_pose_features:
            self.pose_estimator.fit()

    def predict(self, features):
        anomaly_scores = torch.zeros(list(features.values())[0].shape[0]).to(list(features.values())[0].device)
        if self.use_velocity_features:
            anomaly_scores += self.velocity_estimator.predict(features[FeatureType.VELOCITY])
        if self.use_appearance_features:
            anomaly_scores += self.appearance_estimator.predict(features[FeatureType.APPEARANCE])
        if self.use_pose_features:
            anomaly_scores += self.pose_estimator.predict(features[FeatureType.POSE])
        return anomaly_scores


class GroupedKNNEstimator(BaseDensityEstimator):

    def __init__(self, n_neighbors: int) -> None:
        super().__init__()

        self.n_neighbors = n_neighbors
        self.memory_bank: dict[Any, list[Tensor] | Tensor] = {}
        self.normalization_statistics = MinMax()

    def update(self, features: Tensor, group: Any = None):
        """
        Args: features [1, N, M]v or [N, M]
        """
        group = group or "default"

        if group in self.memory_bank:
            self.memory_bank[group].append(features)
        else:
            self.memory_bank[group] = [features]

    def fit(self):
        self.memory_bank = {key: torch.vstack(value) for key, value in self.memory_bank.items()}
        self._compute_normalization_statistics()

    def predict(self, features: Tensor, group: Any = None, n_neighbors: int | None = None, normalize: bool = True):

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
    def _nearest_neighbors(memory_bank, features, n_neighbors: int = 1):

        distances = torch.cdist(features, memory_bank, p=2.0)  # euclidean norm
        if n_neighbors == 1:
            # when n_neighbors is 1, speed up computation by using min instead of topk
            patch_scores, _ = distances.min(1)
            return patch_scores.unsqueeze(1)
        else:
            patch_scores, _ = distances.topk(k=n_neighbors, largest=False, dim=1)
        return patch_scores

    def _compute_normalization_statistics(self):

        for group, features in self.memory_bank.items():
            nns = self.predict(features, group, normalize=False)
            self.normalization_statistics.update(nns)

        self.normalization_statistics.compute()

    def _normalize(self, distances):

        return (distances - self.normalization_statistics.min) / (self.normalization_statistics.max - self.normalization_statistics.min)


class GMMEstimator(BaseDensityEstimator):

    def __init__(self, n_components: int = 2) -> None:
        super().__init__()

        self.gmm = GaussianMixture(n_components=n_components, random_state=0)
        self.memory_bank: list[Tensor] | Tensor = []

        self.normalization_statistics = MinMax()

    def update(self, features):
        self.memory_bank.append(features)

    def fit(self):
        self.memory_bank = torch.vstack(self.memory_bank)
        self.gmm.fit(self.memory_bank.cpu())
        self._compute_normalization_statistics()

    def predict(self, features, normalize: bool = True):
        density = -self.gmm.score_samples(features.cpu())
        density = Tensor(density).to(self.normalization_statistics.device)
        if normalize:
            density = (density - self.normalization_statistics.min) / (self.normalization_statistics.max - self.normalization_statistics.min)
        return density

    def _compute_normalization_statistics(self):
        training_scores = self.predict(self.memory_bank, normalize=False)
        self.normalization_statistics.update(training_scores)
        self.normalization_statistics.compute()
