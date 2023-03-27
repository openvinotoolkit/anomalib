from __future__ import annotations

import clip
import torch
from torch import nn, Tensor
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchvision.ops import roi_align
from sklearn.mixture import GaussianMixture

from anomalib.utils.metrics.min_max import MinMax


class AiVadModel(nn.Module):
    def __init__(self):
        super().__init__()
        # initialize flow extractor
        self.flow_extractor = raft_small(weights=Raft_Small_Weights)
        # initialize region extractor
        self.region_extractor = keypointrcnn_resnet50_fpn(
            weights=KeypointRCNN_ResNet50_FPN_Weights, box_score_thresh=0.8
        )
        # initialize feature extractor
        self.encoder, _ = clip.load("ViT-B/16")

        # define mem banks
        self.velocity_embeddings: Tensor
        self.pose_embeddings: Tensor
        self.feature_embeddings: Tensor

        # define gmm
        self.velocity_estimator = GaussianMixture(n_components=2, random_state=0)

        # norm stats
        self.pose_norm = MinMax()
        self.feature_norm = MinMax()
        self.velocity_norm = MinMax()

    def forward(self, batch):
        self.flow_extractor.eval()
        self.region_extractor.eval()
        self.encoder.eval()

        batch_size = batch.shape[0]

        # 1. compute optical flow and extract regions
        first_frame = batch[:, 0, ...]
        last_frame = batch[:, -1, ...]
        with torch.no_grad():
            flow = self.flow_extractor(first_frame, last_frame)[
                -1
            ]  # only interested in output of last iteration of flow estimation
            regions = self.region_extractor(last_frame)
        # convert from list of [N, 4] tensors to single [N, 5] tensor where each row is [index-in-batch, x1, y1, x2, y2]
        boxes_list = [batch_item["boxes"] for batch_item in regions]
        indices = torch.repeat_interleave(
            torch.arange(len(regions)), Tensor([boxes.shape[0] for boxes in boxes_list]).int()
        )
        boxes = torch.cat([indices.unsqueeze(1).to(batch.device), torch.cat(boxes_list)], dim=1)

        flow_regions = roi_align(flow, boxes, output_size=[224, 224])
        rgb_regions = roi_align(last_frame, boxes, output_size=[224, 224])

        # 2. extract velocity
        velocity = extract_velocity(flow_regions)
        velocity = [velocity[indices == i] for i in range(batch_size)]  # convert back to list

        # 3. extract pose
        poses = extract_poses(regions)
        poses = [pos.reshape(pos.shape[0], -1) for pos in poses]

        # 4. CLIP
        rgb_regions = [rgb_regions[indices == i] for i in range(batch_size)]  # convert back to list
        features = []
        for regions in rgb_regions:
            batched_regions = torch.split(regions, batch_size)
            with torch.no_grad():
                features.append(torch.vstack([self.encoder.encode_image(batch) for batch in batched_regions]))

        if self.training:
            # return features
            return velocity, poses, features

        # infer
        velocity_scores = [self.velocity_estimator.score_samples(vel) for vel in velocity]
        pose_scores = [
            nearest_neighbors(torch.vstack(self.pose_embeddings), pos.reshape(pos.shape[0], -1).cpu(), 9)
            for pos in poses
        ]
        appearance_scores = [
            nearest_neighbors(torch.vstack(self.feature_embeddings).float(), feat.float().cpu(), 9) for feat in features
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
        for velocity, pose, appearance in zip(velocity_scores, pose_scores, appearance_scores):
            anomaly_scores.append(torch.vstack([velocity, pose, appearance]).max(axis=0).values)

        return boxes_list, anomaly_scores

    def compute_normalization_statistics(self):
        for i in range(len(self.pose_embeddings)):
            pose_bank = torch.vstack(self.pose_embeddings[:i] + self.pose_embeddings[i + 1 :])
            pose_embedding = self.pose_embeddings[i]
            nns = nearest_neighbors(pose_bank, pose_embedding, 9)
            self.pose_norm.update(nns)

        for i in range(len(self.feature_embeddings)):
            feature_bank = torch.vstack(self.feature_embeddings[:i] + self.feature_embeddings[i + 1 :]).float()
            feature_embedding = self.feature_embeddings[i].float()
            nns = nearest_neighbors(feature_bank, feature_embedding, 9)
            self.feature_norm.update(nns)

        velocity_training_scores = self.velocity_estimator.score_samples(self.velocity_embeddings)
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
    else:
        patch_scores, _ = distances.topk(k=n_neighbors, largest=False, dim=1)
    return patch_scores.mean(axis=1)


def extract_velocity(flows):
    # cartesian to polar
    # mag = torch.linalg.norm(flow, axis=0, ord=2)
    mag_batch = torch.linalg.norm(flows, axis=1, ord=2)
    # theta = torch.atan2(flow[0], flow[1])
    theta_batch = torch.atan2(flows[:, 0, ...], flows[:, 1, ...])

    # compute velocity histogram
    n_bins = 8
    velocity_hictograms = []
    for mag, theta in zip(mag_batch, theta_batch):
        histogram = torch.histogram(
            input=theta.cpu(), bins=n_bins, range=(-torch.pi, torch.pi), weight=mag.cpu() / mag.sum().cpu()
        )
        velocity_hictograms.append(histogram.hist)
    return torch.stack(velocity_hictograms)


def extract_poses(regions):
    poses = []
    for region in regions:
        boxes = region["boxes"].unsqueeze(1)
        keypoints = region["keypoints"]
        normalized_keypoints = (keypoints[..., :2] - boxes[..., :2]) / (boxes[..., 2:] - boxes[..., :2])
        poses.append(normalized_keypoints)
    return poses
