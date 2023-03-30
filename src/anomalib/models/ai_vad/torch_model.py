from __future__ import annotations

import torch
from torch import nn, Tensor

from anomalib.utils.metrics.min_max import MinMax
from anomalib.models.ai_vad.flow import FlowExtractor
from anomalib.models.ai_vad.regions import RegionExtractor
from anomalib.models.ai_vad.features import FeatureExtractor

from anomalib.models.ai_vad.density import CombinedDensityEstimator


class AiVadModel(nn.Module):
    def __init__(self):
        super().__init__()
        # initialize flow extractor
        self.flow_extractor = FlowExtractor()
        # initialize region extractor
        self.region_extractor = RegionExtractor()
        # initialize feature extractor
        self.feature_extractor = FeatureExtractor()

        self.density_estimator = CombinedDensityEstimator(
            use_pose_features=False,
            use_velocity_features=True,
            use_appearance_features=False,
            n_neighbors_pose=1,
            n_neighbors_appearance=1,
            n_components_velocity=2,
        )

    def forward(self, batch):
        self.flow_extractor.eval()
        self.region_extractor.eval()
        self.feature_extractor.eval()

        # 1. get first and last frame from clip
        first_frame = batch[:, 0, ...]
        last_frame = batch[:, -1, ...]

        # 2. extract flows and regions
        with torch.no_grad():
            flows = self.flow_extractor(first_frame, last_frame)
            regions = self.region_extractor(last_frame)

        # 3. extract pose, appearance and velocity features
        features_per_batch = self.feature_extractor(last_frame, flows, regions)

        if self.training:
            return features_per_batch

        # 4. estimate density
        anomaly_scores = [self.density_estimator(features) for features in features_per_batch]

        boxes_list = [batch_item["boxes"] for batch_item in regions]
        return boxes_list, anomaly_scores
