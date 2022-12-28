"""Torch model for region-based anomaly detection."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Tuple, Union

import torch
from torch import Tensor, nn

from anomalib.models.rkde.density_estimator import DensityEstimator
from anomalib.models.rkde.feature_extractor import FeatureExtractor
from anomalib.models.rkde.region_extractor import RegionExtractor, RoiStage

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
        roi_stage: RoiStage = RoiStage.RCNN,
        roi_score_threshold: float = 0.001,
        max_detections_per_image: int = 100,
        min_box_size: int = 25,
        iou_threshold: float = 0.3,
        n_pca_components: int = 16,
        pre_processing: str = "scale",
        max_training_points: int = 40000,
    ):
        super().__init__()

        self.region_extractor = RegionExtractor(
            stage=roi_stage,
            score_threshold=roi_score_threshold,
            max_detections_per_image=max_detections_per_image,
            min_size=min_box_size,
            iou_threshold=iou_threshold,
        ).eval()

        self.feature_extractor = FeatureExtractor().eval()

        self.density_estimator = DensityEstimator(
            n_pca_components=n_pca_components,
            pre_processing=pre_processing,
            max_training_points=max_training_points,
        )

    def forward(self, batch: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Prediction by normality model.

        Args:
            input (Tensor): Input images.

        Returns:
            Tensor: Predictions
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
        scores = self.density_estimator(features)

        return rois, scores
