from __future__ import annotations

import clip
import torch
from torch import nn, Tensor
from torchvision.ops import roi_align
from torchvision.transforms import Normalize


class FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.appearance_extractor = AppearanceExtractor()
        self.velocity_extractor = VelocityExtractor()
        self.pose_extractor = PoseExtractor()

    def forward(self, rgb_batch, flow_batch, regions):
        batch_size, _channels, _height, _width = rgb_batch.shape

        # convert from list of [N, 4] tensors to single [N, 5] tensor where each row is [index-in-batch, x1, y1, x2, y2]
        boxes_list = [batch_item["boxes"] for batch_item in regions]
        indices = torch.repeat_interleave(
            torch.arange(len(regions)), Tensor([boxes.shape[0] for boxes in boxes_list]).int()
        )
        boxes = torch.cat([indices.unsqueeze(1).to(rgb_batch.device), torch.cat(boxes_list)], dim=1)

        velocity_features = self.velocity_extractor(flow_batch, boxes)
        appearance_features = self.appearance_extractor(rgb_batch, boxes, batch_size)
        pose_features = self.pose_extractor(regions)

        # convert back to list
        velocity_features = [velocity_features[indices == i] for i in range(batch_size)]
        appearance_features = [appearance_features[indices == i] for i in range(batch_size)]

        return dict(velocity=velocity_features, appearance=appearance_features, pose=pose_features)


class AppearanceExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder, _ = clip.load("ViT-B/16")
        self.transform = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def forward(self, batch, boxes, batch_size):
        rgb_regions = roi_align(batch, boxes, output_size=[224, 224])

        features = []
        batched_regions = torch.split(rgb_regions, batch_size)
        with torch.no_grad():
            features = torch.vstack([self.encoder.encode_image(self.transform(batch)) for batch in batched_regions])

        return features


class VelocityExtractor(nn.Module):
    def __init__(self, n_bins: int = 8) -> None:
        super().__init__()

        self.n_bins = n_bins

    def forward(self, flows, boxes):
        flow_regions = roi_align(flows, boxes, output_size=[224, 224])

        # cartesian to polar
        mag_batch = torch.linalg.norm(flow_regions, axis=1, ord=2)
        theta_batch = torch.atan2(flow_regions[:, 0, ...], flow_regions[:, 1, ...])

        # compute velocity histogram
        velocity_hictograms = []
        for mag, theta in zip(mag_batch, theta_batch):
            histogram_mag = torch.histogram(
                input=theta.cpu(), bins=self.n_bins, range=(-torch.pi, torch.pi), weight=mag.cpu()
            ).hist
            histogram_counts = torch.histogram(input=theta.cpu(), bins=self.n_bins, range=(-torch.pi, torch.pi)).hist
            final_histogram = torch.zeros_like(histogram_mag)
            mask = histogram_counts != 0
            final_histogram[mask] = histogram_mag[mask] / histogram_counts[mask]
            velocity_hictograms.append(final_histogram)
            # velocity_hictograms.append(histogram_mag)

        return torch.stack(velocity_hictograms)


class PoseExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, regions):
        poses = []
        for region in regions:
            boxes = region["boxes"].unsqueeze(1)
            keypoints = region["keypoints"]
            normalized_keypoints = (keypoints[..., :2] - boxes[..., :2]) / (boxes[..., 2:] - boxes[..., :2])
            poses.append(normalized_keypoints.reshape(normalized_keypoints.shape[0], -1))
        return poses
