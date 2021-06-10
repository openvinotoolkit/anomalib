"""
Normality model of DFKDE
"""

import random
from typing import Optional, Tuple

import torch
import torch.nn as nn

from anomalib.core.model.kde import GaussianKDE
from anomalib.core.model.pca import PCA


class NormalityModel(nn.Module):
    """
    Normality Model for the DFKDE algorithm
    """

    def __init__(
        self,
        n_comps: int = 16,
        pre_processing: str = "scale",
        filter_count: int = 40000,
        threshold_steepness: float = 0.05,
        threshold_offset: float = 12.0,
    ):
        super().__init__()
        self.n_components = n_comps
        self.pre_processing = pre_processing
        self.filter_count = filter_count
        self.threshold_steepness = threshold_steepness
        self.threshold_offset = threshold_offset

        self.pca_model = PCA(n_components=self.n_components)
        self.kde_model = GaussianKDE()

        self.register_buffer("max_length", torch.Tensor(torch.Size([])))
        self.max_length = torch.Tensor(torch.Size([]))

    def fit(self, dataset: torch.Tensor):
        """
        Fit a kde model to dataset

        Args:
            dataset: Input dataset to fit the model.
            dataset: torch.Tensor:

        Returns:
            Boolean confirming whether the training is successful.

        """

        if dataset.shape[0] < self.n_components:
            print("Not enough features to commit. Not making a model.")
            return False

        # if max training points is non-zero and smaller than number of staged features, select random subset
        if self.filter_count and dataset.shape[0] > self.filter_count:
            selected_idx = torch.tensor(random.sample(range(dataset.shape[0]), self.filter_count))
            selected_features = dataset[selected_idx]
        else:
            selected_features = dataset

        feature_stack = self.pca_model.fit_transform(selected_features)
        feature_stack, max_length = self.preprocess(feature_stack)
        self.max_length = max_length
        self.kde_model.fit(feature_stack)

        return True

    def preprocess(self, feature_stack: torch.Tensor, max_length: Optional[int] = None) -> Tuple[torch.Tensor, int]:
        """Pre process the CNN features.

        Args:
          feature_stack: Features extracted from CNN
          max_length:
          feature_stack: torch.Tensor:
          max_length: Optional[int]:  (Default value = None)

        Returns:

        """

        if self.pre_processing == "norm":
            feature_stack /= torch.norm(feature_stack, 2, 1)[:, None]
        elif self.pre_processing == "scale":
            max_length = max_length if max_length else torch.max(torch.norm(feature_stack, 2, 1))
            feature_stack /= max_length
        else:
            raise RuntimeError("Unknown pre-processing mode. Available modes are: Normalized and Scale.")
        return feature_stack, max_length

    def evaluate(
        self, sem_feats: torch.Tensor, as_density: Optional[bool] = False, as_log_likelihood: Optional[bool] = False
    ) -> torch.Tensor:
        """
        Compute the KDE scores

        Args:
            sem_feats:
            as_density:
            as_log_likelihood:

        Returns:

        """

        sem_feats = self.pca_model.transform(sem_feats)
        sem_feats, _ = self.preprocess(sem_feats, self.max_length)
        kde_scores = self.kde_model(sem_feats)

        # add small constant to avoid zero division in log computation
        kde_scores += 1e-300

        score = kde_scores if as_density else 1.0 / kde_scores

        if as_log_likelihood:
            score = torch.log(score)

        return score

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Predicts the probability that the features belong to the anomalous class.

        Args:
          features: Feature from which the output probabilities are detected.
          features: torch.Tensor:

        Returns:
          Detection probabilities

        """

        densities = self.evaluate(features, as_density=True, as_log_likelihood=True)
        probabilities = self.to_probability(densities)

        return probabilities

    def to_probability(self, densities: torch.Tensor) -> torch.Tensor:
        """Converts density scores to anomaly probabilities
        (see https://www.desmos.com/calculator/ifju7eesg7)

        Args:
          densities: density of an image
          densities: torch.Tensor:

        Returns:
          probability that image with {density} is anomalous

        """

        return 1 / (1 + torch.exp(self.threshold_steepness * (densities - self.threshold_offset)))
