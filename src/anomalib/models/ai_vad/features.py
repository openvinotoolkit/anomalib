from __future__ import annotations

from enum import Enum

import clip
import torch
from torch import Tensor, nn
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn
from torchvision.models.detection.roi_heads import keypointrcnn_inference
from torchvision.ops import roi_align
from torchvision.transforms import Normalize


class FeatureType(str, Enum):
    POSE = "pose"
    VELOCITY = "velocity"
    APPEARANCE = "appearance"


class FeatureExtractor(nn.Module):
    def __init__(self, n_velocity_bins: int = 8) -> None:
        super().__init__()

        self.appearance_extractor = AppearanceExtractor()
        self.velocity_extractor = VelocityExtractor(n_bins=n_velocity_bins)
        self.pose_extractor = PoseExtractor()

    def forward(self, rgb_batch, flow_batch, regions):
        batch_size = rgb_batch.shape[0]

        # convert from list of [N, 4] tensors to single [N, 5] tensor where each row is [index-in-batch, x1, y1, x2, y2]
        boxes_list = [batch_item["boxes"] for batch_item in regions]
        indices = torch.repeat_interleave(
            torch.arange(len(regions)), Tensor([boxes.shape[0] for boxes in boxes_list]).int()
        )
        boxes = torch.cat([indices.unsqueeze(1).to(rgb_batch.device), torch.cat(boxes_list)], dim=1)

        velocity_features = self.velocity_extractor(flow_batch, boxes)
        appearance_features = self.appearance_extractor(rgb_batch, boxes, batch_size)
        # pose_features = self.pose_extractor(regions)
        pose_features = self.pose_extractor(rgb_batch, boxes_list)

        # convert back to list
        velocity_features = [velocity_features[indices == i] for i in range(batch_size)]
        appearance_features = [appearance_features[indices == i] for i in range(batch_size)]

        return [
            {FeatureType.VELOCITY: velocity, FeatureType.APPEARANCE: appearance, FeatureType.POSE: pose}
            for velocity, appearance, pose in zip(velocity_features, appearance_features, pose_features)
        ]


class AppearanceExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder, _ = clip.load("ViT-B/16")
        self.transform = Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

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

        return torch.stack(velocity_hictograms).to(flows.device)


class PoseExtractor(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        model = keypointrcnn_resnet50_fpn(weights=weights)
        self.model = model
        self.transform = model.transform
        self.backbone = model.backbone
        self.roi_heads = model.roi_heads

    @staticmethod
    def _post_process(keypoint_detections):
        poses = []
        for detection in keypoint_detections:
            boxes = detection["boxes"].unsqueeze(1)
            keypoints = detection["keypoints"]
            normalized_keypoints = (keypoints[..., :2] - boxes[..., :2]) / (boxes[..., 2:] - boxes[..., :2])
            poses.append(normalized_keypoints.reshape(normalized_keypoints.shape[0], -1))
        return poses

    def forward(self, batch, boxes):
        images, _ = self.transform(batch)
        features = self.backbone(images.tensors)

        image_sizes = [b.shape[-2:] for b in batch]
        scales = [Tensor(new) / Tensor([orig[0], orig[1]]) for orig, new in zip(image_sizes, images.image_sizes)]

        boxes = [box * scale.repeat(2).to(box.device) for box, scale in zip(boxes, scales)]

        keypoint_features = self.roi_heads.keypoint_roi_pool(features, boxes, images.image_sizes)
        keypoint_features = self.roi_heads.keypoint_head(keypoint_features)
        keypoint_logits = self.roi_heads.keypoint_predictor(keypoint_features)
        keypoints_probs, _ = keypointrcnn_inference(keypoint_logits, boxes)

        keypoint_detections = self.transform.postprocess(
            [{"keypoints": kp, "boxes": bx} for kp, bx in zip(keypoints_probs, boxes)], images.image_sizes, image_sizes
        )
        return self._post_process(keypoint_detections)
