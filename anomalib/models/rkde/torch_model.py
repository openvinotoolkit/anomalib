"""Torch model for region-based anomaly detection."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

from anomalib.models.components.classification import (
    FeatureScalingMethod,
    KDEClassifier,
)
from anomalib.models.rkde.feature_extractor import FeatureExtractor
from anomalib.models.rkde.region_extractor import RegionExtractor, RoiStage

logger = logging.getLogger(__name__)


class RkdeModel(nn.Module):
    """Torch Model for the Region-based Anomaly Detection Model.

    Args:
        roi_stage (RoiStage, optional): Processing stage from which rois are extracted.
        roi_score_threshold (float, optional): Mimumum confidence score for the region proposals.
        min_size (int, optional): Minimum size in pixels for the region proposals.
        iou_threshold (float, optional): Intersection-Over-Union threshold used during NMS.
        max_detections_per_image (int, optional): Maximum number of region proposals per image.
        n_pca_components (int, optional): Number of PCA components. Defaults to 16.
        feature_scaling_method (FeatureScalingMethod, optional): Scaling method applied to features before passing to
            KDE. Options are `norm` (normalize to unit vector length) and `scale` (scale to max length observed in
            training).
        max_training_points (int, optional): Maximum number of training points to fit the KDE model. Defaults to 40000.
    """

    def __init__(
        self,
        # roi params
        roi_stage: RoiStage = RoiStage.RCNN,
        roi_score_threshold: float = 0.001,
        min_box_size: int = 25,
        iou_threshold: float = 0.3,
        max_detections_per_image: int = 100,
        # kde params
        n_pca_components: int = 16,
        feature_scaling_method: FeatureScalingMethod = FeatureScalingMethod.SCALE,
        max_training_points: int = 40000,
    ) -> None:
        super().__init__()

        self.region_extractor = RegionExtractor(
            stage=roi_stage,
            score_threshold=roi_score_threshold,
            min_size=min_box_size,
            iou_threshold=iou_threshold,
            max_detections_per_image=max_detections_per_image,
        ).eval()

        self.feature_extractor = FeatureExtractor().eval()

        self.classifier = KDEClassifier(
            n_pca_components=n_pca_components,
            feature_scaling_method=feature_scaling_method,
            max_training_points=max_training_points,
        )

    def fit(self, embeddings: Tensor) -> bool:
        """Fit the model using a set of collected embeddings.

        Args:
            embeddings (Tensor): Input embeddings to fit the model.

        Returns:
            Boolean confirming whether the training is successful.
        """
        return self.classifier.fit(embeddings)

    def forward(self, batch: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """Prediction by normality model.

        Args:
            input (Tensor): Input images.

        Returns:
            Tensor | tuple[Tensor, Tensor]: The extracted features (when in training mode), or the predicted rois
                and corresponding anomaly scores.
        """
        self.region_extractor.eval()
        self.feature_extractor.eval()

        # 1. apply region extraction
        rois = self.region_extractor(batch)

        # 2. apply feature extraction
        if rois.shape[0] == 0:
            # cannot extract features when no rois are retrieved
            features = torch.empty((0, 4096)).to(batch.device)
        else:
            features = self.feature_extractor(batch, rois.clone())

        if self.training:
            return features

        # 3. apply density estimation
        scores = self.classifier(features)

        return rois, scores
