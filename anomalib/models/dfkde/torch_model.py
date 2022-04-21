"""Normality model of DFKDE."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import logging
import random
from typing import List, Optional, Tuple

import torch
import torchvision
from torch import Tensor, nn

from anomalib.models.components import PCA, FeatureExtractor, GaussianKDE

logger = logging.getLogger(__name__)


class DfkdeModel(nn.Module):
    """Normality Model for the DFKDE algorithm.

    Args:
        backbone (str): Pre-trained model backbone.
        n_comps (int, optional): Number of PCA components. Defaults to 16.
        pre_processing (str, optional): Preprocess features before passing to KDE.
            Options are between `norm` and `scale`. Defaults to "scale".
        filter_count (int, optional): Number of training points to fit the KDE model. Defaults to 40000.
        threshold_steepness (float, optional): Controls how quickly the value saturates around zero. Defaults to 0.05.
        threshold_offset (float, optional): Offset of the density function from 0. Defaults to 12.0.
    """

    def __init__(
        self,
        backbone: str,
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

        _backbone = getattr(torchvision.models, backbone)
        self.feature_extractor = FeatureExtractor(backbone=_backbone(pretrained=True), layers=["avgpool"]).eval()

        self.pca_model = PCA(n_components=self.n_components)
        self.kde_model = GaussianKDE()

        self.register_buffer("max_length", Tensor(torch.Size([])))
        self.max_length = Tensor(torch.Size([]))

    def get_features(self, batch: Tensor) -> Tensor:
        """Extract features from the pretrained network.

        Args:
            batch (Tensor): Image batch.

        Returns:
            Tensor: Tensor containing extracted features.
        """
        self.feature_extractor.eval()
        layer_outputs = self.feature_extractor(batch)
        layer_outputs = torch.cat(list(layer_outputs.values())).detach()
        return layer_outputs

    def fit(self, embeddings: List[Tensor]) -> bool:
        """Fit a kde model to embeddings.

        Args:
            embeddings (Tensor): Input embeddings to fit the model.

        Returns:
            Boolean confirming whether the training is successful.
        """
        _embeddings = torch.vstack(embeddings)

        if _embeddings.shape[0] < self.n_components:
            logger.info("Not enough features to commit. Not making a model.")
            return False

        # if max training points is non-zero and smaller than number of staged features, select random subset
        if self.filter_count and _embeddings.shape[0] > self.filter_count:
            # pylint: disable=not-callable
            selected_idx = torch.tensor(random.sample(range(_embeddings.shape[0]), self.filter_count))
            selected_features = _embeddings[selected_idx]
        else:
            selected_features = _embeddings

        feature_stack = self.pca_model.fit_transform(selected_features)
        feature_stack, max_length = self.preprocess(feature_stack)
        self.max_length = max_length
        self.kde_model.fit(feature_stack)

        return True

    def preprocess(self, feature_stack: Tensor, max_length: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Pre-process the CNN features.

        Args:
          feature_stack (Tensor): Features extracted from CNN
          max_length (Optional[Tensor]): Used to unit normalize the feature_stack vector. If ``max_len`` is not
            provided, the length is calculated from the ``feature_stack``. Defaults to None.

        Returns:
            (Tuple): Stacked features and length
        """

        if max_length is None:
            max_length = torch.max(torch.linalg.norm(feature_stack, ord=2, dim=1))

        if self.pre_processing == "norm":
            feature_stack /= torch.linalg.norm(feature_stack, ord=2, dim=1)[:, None]
        elif self.pre_processing == "scale":
            feature_stack /= max_length
        else:
            raise RuntimeError("Unknown pre-processing mode. Available modes are: Normalized and Scale.")
        return feature_stack, max_length

    def evaluate(self, features: Tensor, as_log_likelihood: Optional[bool] = False) -> Tensor:
        """Compute the KDE scores.

        The scores calculated from the KDE model are converted to densities. If `as_log_likelihood` is set to true then
            the log of the scores are calculated.

        Args:
            features (Tensor): Features to which the PCA model is fit.
            as_log_likelihood (Optional[bool], optional): If true, gets log likelihood scores. Defaults to False.

        Returns:
            (Tensor): Score
        """

        features = self.pca_model.transform(features)
        features, _ = self.preprocess(features, self.max_length)
        # Scores are always assumed to be passed as a density
        kde_scores = self.kde_model(features)

        # add small constant to avoid zero division in log computation
        kde_scores += 1e-300

        if as_log_likelihood:
            kde_scores = torch.log(kde_scores)

        return kde_scores

    def predict(self, features: Tensor) -> Tensor:
        """Predicts the probability that the features belong to the anomalous class.

        Args:
          features (Tensor): Feature from which the output probabilities are detected.

        Returns:
          Detection probabilities
        """

        densities = self.evaluate(features, as_log_likelihood=True)
        probabilities = self.to_probability(densities)

        return probabilities

    def to_probability(self, densities: Tensor) -> Tensor:
        """Converts density scores to anomaly probabilities (see https://www.desmos.com/calculator/ifju7eesg7).

        Args:
          densities (Tensor): density of an image.

        Returns:
          probability that image with {density} is anomalous
        """

        return 1 / (1 + torch.exp(self.threshold_steepness * (densities - self.threshold_offset)))

    def forward(self, batch: Tensor) -> Tensor:
        """Prediction by normality model.

        Args:
            batch (Tensor): Input images.

        Returns:
            Tensor: Predictions
        """

        feature_vector = self.get_features(batch)
        return self.predict(feature_vector.view(feature_vector.shape[:2]))
