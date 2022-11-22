"""Normality model of DFKDE."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import random
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from anomalib.models.components import PCA, GaussianKDE
from anomalib.models.rkde.feature_extractor import FeatureExtractor, RegionExtractor

logger = logging.getLogger(__name__)


class RkdeModel(nn.Module):
    """Torch Model for the Region-based Anomaly Detection Model.

    Args:
        backbone (str): Pre-trained model backbone.
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
        n_comps (int, optional): Number of PCA components. Defaults to 16.
        pre_processing (str, optional): Preprocess features before passing to KDE.
            Options are between `norm` and `scale`. Defaults to "scale".
        filter_count (int, optional): Number of training points to fit the KDE model. Defaults to 40000.
        threshold_steepness (float, optional): Controls how quickly the value saturates around zero. Defaults to 0.05.
        threshold_offset (float, optional): Offset of the density function from 0. Defaults to 12.0.
    """

    def __init__(
        self,
        region_extractor_stage: str = "rcnn",
        min_box_size: int = 25,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.3,
        n_pca_components: int = 16,
        pre_processing: str = "scale",
        filter_count: int = 40000,
        threshold_steepness: float = 0.05,
        threshold_offset: float = 12.0,
    ):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.n_pca_components = n_pca_components
        self.pre_processing = pre_processing
        self.filter_count = filter_count
        self.threshold_steepness = threshold_steepness
        self.threshold_offset = threshold_offset

        self.region_extractor = RegionExtractor(
            stage=region_extractor_stage, min_size=min_box_size, iou_threshold=iou_threshold
        ).eval()

        self.feature_extractor = FeatureExtractor().eval()

        # self.feature_extractor = FeatureExtractor(
        # region_extractor_stage=region_extractor_stage,
        # min_box_size=min_box_size,
        # iou_threshold=iou_threshold,
        # )

        self.pca_model = PCA(n_components=self.n_pca_components)
        self.kde_model = GaussianKDE()

        self.register_buffer("max_length", Tensor(torch.Size([])))
        self.max_length = Tensor(torch.Size([]))

    def pre_process(self, feature_stack: Tensor, max_length: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
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

    def fit(self, embeddings: List[Tensor]) -> bool:
        """Fit a kde model to embeddings.

        Args:
            embeddings (Tensor): Input embeddings to fit the model.

        Returns:
            Boolean confirming whether the training is successful.
        """
        _embeddings = torch.vstack(embeddings)

        if _embeddings.shape[0] < self.n_pca_components:
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
        feature_stack, max_length = self.pre_process(feature_stack)
        self.max_length = max_length
        self.kde_model.fit(feature_stack)

        return True

    def compute_kde_scores(self, features: Tensor, as_log_likelihood: Optional[bool] = False) -> Tensor:
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
        features, _ = self.pre_process(features, self.max_length)
        # Scores are always assumed to be passed as a density
        kde_scores = self.kde_model(features)

        # add small constant to avoid zero division in log computation
        kde_scores += 1e-300

        if as_log_likelihood:
            kde_scores = torch.log(kde_scores)

        return kde_scores

    def compute_probabilities(self, scores: Tensor) -> Tensor:
        """Converts density scores to anomaly probabilities (see https://www.desmos.com/calculator/ifju7eesg7).

        Args:
          scores (Tensor): density of an image.

        Returns:
          probability that image with {density} is anomalous
        """

        return 1 / (1 + torch.exp(self.threshold_steepness * (scores - self.threshold_offset)))

    def predict(self, features: Tensor, batch_rois: List[Tensor]) -> Tensor:
        """Predicts the probability that the features belong to the anomalous class.

        Args:
          features (Tensor): Feature from which the output probabilities are detected.
          rois (Tensor): RoIs from which the features are extracted.

        Returns:
          Detection probabilities
        """

        scores = self.compute_kde_scores(features, as_log_likelihood=True)
        scores = self.compute_probabilities(scores)
        batch_scores = torch.split(scores, [len(rois) for rois in batch_rois])

        # remove items with low probability
        batch_keep = [scores > self.confidence_threshold for scores in batch_scores]
        batch_rois = [rois[keep] for rois, keep in zip(batch_rois, batch_keep)]
        batch_scores = [scores[keep] for scores, keep in zip(batch_scores, batch_keep)]

        # keep = probabilities > self.confidence_threshold
        # rois = rois[keep]
        # probabilities = probabilities[keep]

        # TODO: Here we need to sort out how to handle box detections.

        return batch_rois, batch_scores

    def forward(self, input: Tensor) -> Tensor:
        """Prediction by normality model.

        Args:
            input (Tensor): Input images.

        Returns:
            Tensor: Predictions
        """
        self.region_extractor.eval()
        self.feature_extractor.eval()

        batch_rois = self.region_extractor(input)

        if all(len(rois) == 0 for rois in batch_rois):
            # no rois found in this batch
            features = torch.empty((0, 4096)).to(input.device)
        else:
            features = self.feature_extractor(input, batch_rois)

        if self.training:
            return features
        return self.predict(features, batch_rois)
