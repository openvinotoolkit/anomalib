"""Feature extraction module for AI-VAD model implementation."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

import torch
from torch import nn
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn
from torchvision.models.detection.roi_heads import keypointrcnn_inference
from torchvision.ops import roi_align
from torchvision.transforms import Normalize

from .clip import clip


class FeatureType(str, Enum):
    """Names of the different feature streams used in AI-VAD."""

    POSE = "pose"
    VELOCITY = "velocity"
    DEEP = "deep"


class FeatureExtractor(nn.Module):
    """Feature extractor for AI-VAD.

    Args:
        n_velocity_bins (int): Number of discrete bins used for velocity histogram features.
            Defaults to ``8``.
        use_velocity_features (bool): Flag indicating if velocity features should be used.
            Defaults to ``True``.
        use_pose_features (bool): Flag indicating if pose features should be used.
            Defaults to ``True``.
        use_deep_features (bool): Flag indicating if deep features should be used.
            Defaults to ``True``.
    """

    def __init__(
        self,
        n_velocity_bins: int = 8,
        use_velocity_features: bool = True,
        use_pose_features: bool = True,
        use_deep_features: bool = True,
    ) -> None:
        super().__init__()
        if not (use_velocity_features or use_pose_features or use_deep_features):
            msg = "At least one feature stream must be enabled."
            raise ValueError(msg)

        self.use_velocity_features = use_velocity_features
        self.use_pose_features = use_pose_features
        self.use_deep_features = use_deep_features

        self.deep_extractor = DeepExtractor()
        self.velocity_extractor = VelocityExtractor(n_bins=n_velocity_bins)
        self.pose_extractor = PoseExtractor()

    def forward(
        self,
        rgb_batch: torch.Tensor,
        flow_batch: torch.Tensor,
        regions: list[dict],
    ) -> list[dict]:
        """Forward pass through the feature extractor.

        Extract any combination of velocity, pose and deep features depending on configuration.

        Args:
            rgb_batch (torch.Tensor): Batch of RGB images of shape (N, 3, H, W)
            flow_batch (torch.Tensor): Batch of optical flow images of shape (N, 2, H, W)
            regions (list[dict]): Region information per image in batch.

        Returns:
            list[dict]: Feature dictionary per image in batch.
        """
        batch_size = rgb_batch.shape[0]

        # convert from list of [N, 4] tensors to single [N, 5] tensor where each row is [index-in-batch, x1, y1, x2, y2]
        boxes_list = [batch_item["boxes"] for batch_item in regions]
        indices = torch.repeat_interleave(
            torch.arange(len(regions)),
            torch.Tensor([boxes.shape[0] for boxes in boxes_list]).int(),
        )
        boxes = torch.cat([indices.unsqueeze(1).to(rgb_batch.device), torch.cat(boxes_list)], dim=1)

        # Extract features
        feature_dict = {}
        if self.use_velocity_features:
            velocity_features = self.velocity_extractor(flow_batch, boxes)
            feature_dict[FeatureType.VELOCITY] = [velocity_features[indices == i] for i in range(batch_size)]
        if self.use_pose_features:
            pose_features = self.pose_extractor(rgb_batch, boxes_list)
            feature_dict[FeatureType.POSE] = pose_features
        if self.use_deep_features:
            deep_features = self.deep_extractor(rgb_batch, boxes, batch_size)
            feature_dict[FeatureType.DEEP] = [deep_features[indices == i] for i in range(batch_size)]

        # dict of lists to list of dicts
        return [dict(zip(feature_dict, item, strict=True)) for item in zip(*feature_dict.values(), strict=True)]


class DeepExtractor(nn.Module):
    """Deep feature extractor.

    Extracts the deep (appearance) features from the input regions.
    """

    def __init__(self) -> None:
        super().__init__()

        self.encoder, _ = clip.load("ViT-B/16")
        self.transform = Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.output_dim = self.encoder.visual.output_dim

    def forward(self, batch: torch.Tensor, boxes: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Extract deep features using CLIP encoder.

        Args:
            batch (torch.Tensor): Batch of RGB input images of shape (N, 3, H, W)
            boxes (torch.Tensor): Bounding box coordinates of shaspe (M, 5).
                First column indicates batch index of the bbox.
            batch_size (int): Number of images in the batch.

        Returns:
            Tensor: Deep feature tensor of shape (M, 512)
        """
        rgb_regions = roi_align(batch, boxes, output_size=[224, 224])

        batched_regions = torch.split(rgb_regions, batch_size)
        batched_regions = [batch for batch in batched_regions if batch.numel() != 0]
        with torch.no_grad():
            features = [self.encoder.encode_image(self.transform(batch)) for batch in batched_regions]
            return torch.vstack(features).float() if len(features) else torch.empty(0, self.output_dim).to(batch.device)


class VelocityExtractor(nn.Module):
    """Velocity feature extractor.

    Extracts histograms of optical flow magnitude and direction.

    Args:
        n_bins (int): Number of direction bins used for the feature histograms.
    """

    def __init__(self, n_bins: int = 8) -> None:
        super().__init__()

        self.n_bins = n_bins

    def forward(self, flows: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """Extract velocioty features by filling a histogram.

        Args:
            flows (torch.Tensor): Batch of optical flow images of shape (N, 2, H, W)
            boxes (torch.Tensor): Bounding box coordinates of shaspe (M, 5).
                First column indicates batch index of the bbox.

        Returns:
            Tensor: Velocity feature tensor of shape (M, n_bins)
        """
        flow_regions = roi_align(flows, boxes, output_size=[224, 224])

        # cartesian to polar
        mag_batch = torch.linalg.norm(flow_regions, axis=1, ord=2)
        theta_batch = torch.atan2(flow_regions[:, 0, ...], flow_regions[:, 1, ...])

        # compute velocity histogram
        velocity_histograms = []
        for mag, theta in zip(mag_batch, theta_batch, strict=True):
            histogram_mag = torch.histogram(
                input=theta.cpu(),
                bins=self.n_bins,
                range=(-torch.pi, torch.pi),
                weight=mag.cpu(),
            ).hist
            histogram_counts = torch.histogram(input=theta.cpu(), bins=self.n_bins, range=(-torch.pi, torch.pi)).hist
            final_histogram = torch.zeros_like(histogram_mag)
            mask = histogram_counts != 0
            final_histogram[mask] = histogram_mag[mask] / histogram_counts[mask]
            velocity_histograms.append(final_histogram)

        if len(velocity_histograms) == 0:
            return torch.empty(0, self.n_bins).to(flows.device)
        return torch.stack(velocity_histograms).to(flows.device)


class PoseExtractor(nn.Module):
    """Pose feature extractor.

    Extracts pose features based on estimated body landmark keypoints.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        model = keypointrcnn_resnet50_fpn(weights=weights)
        self.model = model
        self.transform = model.transform
        self.backbone = model.backbone
        self.roi_heads = model.roi_heads

    @staticmethod
    def _post_process(keypoint_detections: list[dict]) -> list[torch.Tensor]:
        """Convert keypoint predictions to 1D feature vectors.

        Post-processing consists of flattening and normalizing to bbox coordinates.

        Args:
            keypoint_detections (list[dict]): Outputs of the keypoint extractor

        Returns:
            list[torch.Tensor]: List of pose feature tensors for each image
        """
        poses = []
        for detection in keypoint_detections:
            boxes = detection["boxes"].unsqueeze(1)
            keypoints = detection["keypoints"]
            normalized_keypoints = (keypoints[..., :2] - boxes[..., :2]) / (boxes[..., 2:] - boxes[..., :2])
            length = normalized_keypoints.shape[-1] * normalized_keypoints.shape[-2]
            poses.append(normalized_keypoints.reshape(normalized_keypoints.shape[0], length))
        return poses

    def forward(self, batch: torch.Tensor, boxes: torch.Tensor) -> list[torch.Tensor]:
        """Extract pose features using a human keypoint estimation model.

        Args:
            batch (torch.Tensor): Batch of RGB input images of shape (N, 3, H, W)
            boxes (torch.Tensor): Bounding box coordinates of shaspe (M, 5).
                First column indicates batch index of the bbox.

        Returns:
            list[torch.Tensor]: list of pose feature tensors for each image.
        """
        images, _ = self.transform(batch)
        features = self.backbone(images.tensors)

        image_sizes = [b.shape[-2:] for b in batch]
        scales = [
            torch.Tensor(new) / torch.Tensor([orig[0], orig[1]])
            for orig, new in zip(image_sizes, images.image_sizes, strict=True)
        ]

        boxes = [box * scale.repeat(2).to(box.device) for box, scale in zip(boxes, scales, strict=True)]

        keypoint_features = self.roi_heads.keypoint_roi_pool(features, boxes, images.image_sizes)
        keypoint_features = self.roi_heads.keypoint_head(keypoint_features)
        keypoint_logits = self.roi_heads.keypoint_predictor(keypoint_features)
        keypoints_probs, _ = keypointrcnn_inference(keypoint_logits, boxes)

        keypoint_detections = self.transform.postprocess(
            [{"keypoints": keypoints, "boxes": box} for keypoints, box in zip(keypoints_probs, boxes, strict=True)],
            images.image_sizes,
            image_sizes,
        )
        return self._post_process(keypoint_detections)
