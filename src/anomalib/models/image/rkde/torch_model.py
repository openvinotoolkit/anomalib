"""Torch model for region-based anomaly detection."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import torch
from torch import nn

from anomalib.data import InferenceBatch
from anomalib.models.components.classification import FeatureScalingMethod, KDEClassifier

from .feature_extractor import FeatureExtractor
from .region_extractor import RegionExtractor

logger = logging.getLogger(__name__)


class RkdeModel(nn.Module):
    """Torch Model for the Region-based Anomaly Detection Model.

    Args:
        box_score_threshold (float, optional): Minimum confidence score for the region proposals.
            Defaults to ``0.001``.
        min_box_size (int, optional): Minimum size in pixels for the region proposals.
            Defaults to ``100``.
        iou_threshold (float, optional): Intersection-Over-Union threshold used during NMS.
            Defaults to ``0.3``.
        max_detections_per_image (int, optional): Maximum number of region proposals per image.
            Defaults to ``100``.
        n_pca_components (int, optional): Number of PCA components.
            Defaults to ``16``.
        feature_scaling_method (FeatureScalingMethod, optional): Scaling method applied to features before passing to
            KDE. Options are `norm` (normalize to unit vector length) and `scale` (scale to max length observed in
            training).
            Defaults to ``FeatureScalingMethod.SCALE``.
        max_training_points (int, optional): Maximum number of training points to fit the KDE model.
            Defaults to ``40000``.
    """

    def __init__(
        self,
        # roi params
        box_score_threshold: float = 0.001,
        min_box_size: int = 100,
        iou_threshold: float = 0.3,
        max_detections_per_image: int = 100,
        # kde params
        n_pca_components: int = 16,
        feature_scaling_method: FeatureScalingMethod = FeatureScalingMethod.SCALE,
        max_training_points: int = 40000,
    ) -> None:
        super().__init__()

        self.region_extractor = RegionExtractor(
            box_score_threshold=box_score_threshold,
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

    def fit(self, embeddings: torch.Tensor) -> bool:
        """Fit the model using a set of collected embeddings.

        Args:
            embeddings (torch.Tensor): Input embeddings to fit the model.

        Returns:
            Boolean confirming whether the training is successful.
        """
        return self.classifier.fit(embeddings)

    def forward(self, batch: torch.Tensor) -> torch.Tensor | InferenceBatch:
        """Prediction by normality model.

        Args:
            batch (torch.Tensor): Input images.

        Returns:
            Tensor | tuple[torch.Tensor, torch.Tensor]: The extracted features (when in training mode),
                or the predicted rois and corresponding anomaly scores.
        """
        self.region_extractor.eval()
        self.feature_extractor.eval()

        # 1. apply region extraction
        with torch.no_grad():
            regions: list[dict[str, torch.Tensor]] = self.region_extractor(batch)

        # convert from list of [N, 4] tensors to single [N, 5] tensor where each row is [index-in-batch, x1, y1, x2, y2]
        boxes_list = [batch_item["boxes"] for batch_item in regions]
        indices = torch.repeat_interleave(
            torch.arange(len(regions)),
            torch.Tensor([boxes.shape[0] for boxes in boxes_list]).int(),
        )
        rois = torch.cat([indices.unsqueeze(1).to(batch.device), torch.cat(boxes_list)], dim=1)

        # 2. apply feature extraction
        if rois.shape[0] == 0:
            # cannot extract features when no rois are retrieved
            features = torch.empty((0, 4096)).to(batch.device)
        else:
            with torch.no_grad():
                features = self.feature_extractor(batch, rois)

        if self.training:
            return features

        # 3. apply density estimation
        scores = self.classifier(features)

        # 4. Compute anomaly map
        masks = torch.cat([region["masks"] for region in regions])
        # Select the mask with the highest score for each region
        anomaly_map = torch.stack(
            [
                torch.amax(masks[indices == i] * scores[indices == i].view(-1, 1, 1, 1), dim=0)
                if i in indices
                else torch.zeros_like(masks[0])
                for i in range(len(regions))
            ],
        )

        # 5. Compute box scores
        pred_scores = torch.stack([
            torch.amax(scores[indices == i]) if i in indices else torch.tensor(0.0, device=scores[0].device)
            for i in range(len(regions))
        ])

        return InferenceBatch(
            pred_score=pred_scores,
            anomaly_map=anomaly_map,
        )
