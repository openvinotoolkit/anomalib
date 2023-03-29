from __future__ import annotations

import torch
from torch import nn, Tensor
from sklearn.mixture import GaussianMixture

from anomalib.utils.metrics.min_max import MinMax
from anomalib.models.ai_vad.flow import FlowExtractor, VelocityExtractor
from anomalib.models.ai_vad.region_extractor import RegionExtractor
from anomalib.models.ai_vad.feature_extractor import FeatureExtractor


class AiVadModel(nn.Module):
    def __init__(self):
        super().__init__()
        # initialize flow extractor
        self.flow_extractor = FlowExtractor()
        self.velocity_extractor = VelocityExtractor(n_bins=8)
        # initialize region extractor
        self.region_extractor = RegionExtractor()
        # initialize feature extractor
        self.feature_extractor = FeatureExtractor()

        # define mem banks
        self.velocity_embeddings: Tensor
        self.pose_embeddings: Tensor
        self.feature_embeddings: Tensor

        # define gmm
        self.velocity_estimator = GaussianMixture(n_components=5, random_state=0)

        # norm stats
        self.pose_norm = MinMax()
        self.feature_norm = MinMax()
        self.velocity_norm = MinMax()

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

        features = self.feature_extractor(last_frame, flows, regions)

        if self.training:
            # return features
            return features

        # infer
        velocity_scores = [-self.velocity_estimator.score_samples(vel) for vel in features["velocity"]]
        pose_scores = [nearest_neighbors(torch.vstack(self.pose_embeddings), pos.cpu(), 1) for pos in features["pose"]]
        appearance_scores = [
            nearest_neighbors(torch.vstack(self.feature_embeddings).float(), feat.float().cpu(), 1)
            for feat in features["appearance"]
        ]

        # normalize scores
        self.velocity_norm.cpu()
        velocity_scores = [
            (Tensor(vel) - self.velocity_norm.min) / (self.velocity_norm.max - self.velocity_norm.min)
            for vel in velocity_scores
        ]

        self.pose_norm.cpu()
        pose_scores = [(pos - self.pose_norm.min) / (self.pose_norm.max - self.pose_norm.min) for pos in pose_scores]

        self.feature_norm.cpu()
        appearance_scores = [
            (Tensor(feat) - self.feature_norm.min) / (self.feature_norm.max - self.feature_norm.min)
            for feat in appearance_scores
        ]

        anomaly_scores = []
        # for velocity, pose, appearance in zip(velocity_scores, pose_scores, appearance_scores):
        #     anomaly_scores.append(torch.vstack([velocity, pose]).max(axis=0).values)
        # anomaly_scores = [app + vel + pose for app, vel, pose in zip(appearance_scores, velocity_scores, pose_scores)]
        anomaly_scores = velocity_scores

        boxes_list = [region_batch["boxes"] for region_batch in regions]

        return boxes_list, anomaly_scores

    def compute_normalization_statistics(self):
        for i in range(len(self.pose_embeddings)):
            pose_bank = torch.vstack(self.pose_embeddings[:i] + self.pose_embeddings[i + 1 :])
            pose_embedding = self.pose_embeddings[i]
            nns = nearest_neighbors(pose_bank, pose_embedding, 1)
            self.pose_norm.update(nns)

        for i in range(len(self.feature_embeddings)):
            feature_bank = torch.vstack(self.feature_embeddings[:i] + self.feature_embeddings[i + 1 :]).float()
            feature_embedding = self.feature_embeddings[i].float()
            nns = nearest_neighbors(feature_bank, feature_embedding, 1)
            self.feature_norm.update(nns)

        velocity_training_scores = -self.velocity_estimator.score_samples(self.velocity_embeddings)
        self.velocity_norm.update(Tensor(velocity_training_scores))

        self.pose_norm.compute()
        self.feature_norm.compute()
        self.velocity_norm.compute()


def nearest_neighbors(memory_bank, embedding: Tensor, n_neighbors: int) -> Tensor:
    """Nearest Neighbours using brute force method and euclidean norm.

    Args:
        embedding (Tensor): Features to compare the distance with the memory bank.
        n_neighbors (int): Number of neighbors to look at

    Returns:
        Tensor: Patch scores.
        Tensor: Locations of the nearest neighbor(s).
    """
    distances = torch.cdist(embedding, memory_bank, p=2.0)  # euclidean norm
    if n_neighbors == 1:
        # when n_neighbors is 1, speed up computation by using min instead of topk
        patch_scores, _ = distances.min(1)
        return patch_scores
    else:
        patch_scores, _ = distances.topk(k=n_neighbors, largest=False, dim=1)
    return patch_scores.mean(axis=1)
