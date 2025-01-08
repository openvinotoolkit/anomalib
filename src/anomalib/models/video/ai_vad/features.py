"""Feature extraction module for AI-VAD model implementation.

This module implements the feature extraction stage of the AI-VAD model. It extracts
three types of features from video regions:

- Velocity features: Histogram of optical flow magnitudes
- Pose features: Human keypoint detections using KeypointRCNN
- Deep features: CLIP embeddings of region crops

Example:
    >>> from anomalib.models.video.ai_vad.features import FeatureExtractor
    >>> import torch
    >>> extractor = FeatureExtractor()
    >>> frames = torch.randn(32, 2, 3, 256, 256)  # (N, L, C, H, W)
    >>> flow = torch.randn(32, 2, 256, 256)  # (N, 2, H, W)
    >>> regions = [{"boxes": torch.randn(5, 4)}] * 32  # List of region dicts
    >>> features = extractor(frames, flow, regions)

The module provides the following components:
    - :class:`FeatureType`: Enum of available feature types
    - :class:`FeatureExtractor`: Main class that handles feature extraction
"""

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
    """Names of the different feature streams used in AI-VAD.

    This enum defines the available feature types that can be extracted from video
    regions in the AI-VAD model.

    Attributes:
        POSE: Keypoint features extracted using KeypointRCNN model
        VELOCITY: Histogram features computed from optical flow magnitudes
        DEEP: Visual embedding features extracted using CLIP model

    Example:
        >>> from anomalib.models.video.ai_vad.features import FeatureType
        >>> feature_type = FeatureType.POSE
        >>> feature_type
        <FeatureType.POSE: 'pose'>
        >>> feature_type == "pose"
        True
        >>> feature_type in [FeatureType.POSE, FeatureType.VELOCITY]
        True
    """

    POSE = "pose"
    VELOCITY = "velocity"
    DEEP = "deep"


class FeatureExtractor(nn.Module):
    """Feature extractor for AI-VAD.

    Extracts velocity, pose and deep features from video regions based on the enabled
    feature types.

    Args:
        n_velocity_bins (int, optional): Number of discrete bins used for velocity
            histogram features. Defaults to ``8``.
        use_velocity_features (bool, optional): Flag indicating if velocity features
            should be used. Defaults to ``True``.
        use_pose_features (bool, optional): Flag indicating if pose features should be
            used. Defaults to ``True``.
        use_deep_features (bool, optional): Flag indicating if deep features should be
            used. Defaults to ``True``.

    Raises:
        ValueError: If none of the feature types (velocity, pose, deep) are enabled.

    Example:
        >>> import torch
        >>> from anomalib.models.video.ai_vad.features import FeatureExtractor
        >>> extractor = FeatureExtractor()
        >>> rgb_batch = torch.randn(32, 3, 256, 256)  # (N, C, H, W)
        >>> flow_batch = torch.randn(32, 2, 256, 256)  # (N, 2, H, W)
        >>> regions = [{"boxes": torch.randn(5, 4)}] * 32  # List of region dicts
        >>> features = extractor(rgb_batch, flow_batch, regions)
        >>> # Returns list of dicts with keys: velocity, pose, deep
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

        Extract any combination of velocity, pose and deep features depending on
        configuration.

        Args:
            rgb_batch (torch.Tensor): Batch of RGB images of shape ``(N, 3, H, W)``.
            flow_batch (torch.Tensor): Batch of optical flow images of shape
                ``(N, 2, H, W)``.
            regions (list[dict]): Region information per image in batch. Each dict
                contains bounding boxes of shape ``(M, 4)``.

        Returns:
            list[dict]: Feature dictionary per image in batch. Each dict contains
                the enabled feature types as keys with corresponding feature tensors
                as values.

        Example:
            >>> import torch
            >>> from anomalib.models.video.ai_vad.features import FeatureExtractor
            >>> extractor = FeatureExtractor()
            >>> rgb_batch = torch.randn(32, 3, 256, 256)  # (N, C, H, W)
            >>> flow_batch = torch.randn(32, 2, 256, 256)  # (N, 2, H, W)
            >>> regions = [{"boxes": torch.randn(5, 4)}] * 32  # List of region dicts
            >>> features = extractor(rgb_batch, flow_batch, regions)
            >>> features[0].keys()  # Features for first image
            dict_keys(['velocity', 'pose', 'deep'])
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

    Extracts deep (appearance) features from input regions using a CLIP vision encoder.

    The extractor uses a pre-trained ViT-B/16 CLIP model to encode image regions into
    a 512-dimensional feature space. Input regions are resized to 224x224 and
    normalized using CLIP's default preprocessing.

    Example:
        >>> import torch
        >>> from anomalib.models.video.ai_vad.features import DeepExtractor
        >>> extractor = DeepExtractor()
        >>> batch = torch.randn(32, 3, 256, 256)  # (N, C, H, W)
        >>> boxes = torch.tensor([[0, 10, 20, 50, 60]])  # (M, 5) with batch indices
        >>> features = extractor(batch, boxes, batch_size=32)
        >>> features.shape
        torch.Size([1, 512])
    """

    def __init__(self) -> None:
        super().__init__()

        self.encoder, _ = clip.load("ViT-B/16")
        self.transform = Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.output_dim = self.encoder.visual.output_dim

    def forward(self, batch: torch.Tensor, boxes: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Extract deep features using CLIP encoder.

        Args:
            batch (torch.Tensor): Batch of RGB input images of shape ``(N, 3, H, W)``
            boxes (torch.Tensor): Bounding box coordinates of shape ``(M, 5)``. First
                column indicates batch index of the bbox, remaining columns are
                coordinates ``[x1, y1, x2, y2]``.
            batch_size (int): Number of images in the batch.

        Returns:
            torch.Tensor: Deep feature tensor of shape ``(M, 512)``, where ``M`` is
                the number of input regions and 512 is the CLIP feature dimension.
                Returns empty tensor if no valid regions.
        """
        rgb_regions = roi_align(batch, boxes, output_size=[224, 224])

        batched_regions = torch.split(rgb_regions, batch_size)
        batched_regions = [batch for batch in batched_regions if batch.numel() != 0]
        with torch.no_grad():
            features = [self.encoder.encode_image(self.transform(batch)) for batch in batched_regions]
            return torch.vstack(features).float() if len(features) else torch.empty(0, self.output_dim).to(batch.device)


class VelocityExtractor(nn.Module):
    """Velocity feature extractor.

    Extracts histograms of optical flow magnitude and direction from video regions.
    The histograms capture motion patterns by binning flow vectors based on their
    direction and weighting by magnitude.

    Args:
        n_bins (int, optional): Number of direction bins used for the feature
            histograms. Defaults to ``8``.

    Example:
        >>> import torch
        >>> from anomalib.models.video.ai_vad.features import VelocityExtractor
        >>> extractor = VelocityExtractor(n_bins=8)
        >>> flows = torch.randn(32, 2, 256, 256)  # (N, 2, H, W)
        >>> boxes = torch.tensor([[0, 10, 20, 50, 60]])  # (M, 5) with batch indices
        >>> features = extractor(flows, boxes)
        >>> features.shape
        torch.Size([1, 8])
    """

    def __init__(self, n_bins: int = 8) -> None:
        super().__init__()

        self.n_bins = n_bins

    def forward(self, flows: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """Extract velocity features by computing flow direction histograms.

        For each region, computes a histogram of optical flow directions weighted by
        flow magnitudes. The flow vectors are converted from cartesian to polar
        coordinates, with directions binned into ``n_bins`` equal intervals between
        ``-π`` and ``π``. The histogram values are normalized by the bin counts.

        Args:
            flows (torch.Tensor): Batch of optical flow images of shape
                ``(N, 2, H, W)``, where the second dimension contains x and y flow
                components.
            boxes (torch.Tensor): Bounding box coordinates of shape ``(M, 5)``. First
                column indicates batch index of the bbox, remaining columns are
                coordinates ``[x1, y1, x2, y2]``.

        Returns:
            torch.Tensor: Velocity feature tensor of shape ``(M, n_bins)``, where
                ``M`` is the number of input regions. Returns empty tensor if no
                valid regions.
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

    Extracts pose features based on estimated body landmark keypoints using a
    KeypointRCNN model.

    Example:
        >>> import torch
        >>> from anomalib.models.video.ai_vad.features import PoseExtractor
        >>> extractor = PoseExtractor()
        >>> batch = torch.randn(2, 3, 256, 256)  # (N, C, H, W)
        >>> boxes = torch.tensor([[0, 10, 10, 50, 50], [1, 20, 20, 60, 60]])
        >>> features = extractor(batch, boxes)
        >>> # Returns list of pose feature tensors for each image
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the pose feature extractor.

        Loads a pre-trained KeypointRCNN model and extracts its components for
        feature extraction.
        """
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

        Post-processing consists of flattening the keypoint coordinates and
        normalizing them relative to the bounding box coordinates.

        Args:
            keypoint_detections (list[dict]): Outputs of the keypoint extractor
                containing detected keypoints and bounding boxes.

        Returns:
            list[torch.Tensor]: List of pose feature tensors for each image, where
                each tensor has shape ``(N, K*2)`` with ``N`` being the number of
                detections and ``K`` the number of keypoints.
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

        The method performs the following steps:
        1. Transform input images
        2. Extract backbone features
        3. Pool ROI features for each box
        4. Predict keypoint locations
        5. Post-process predictions

        Args:
            batch (torch.Tensor): Batch of RGB input images of shape
                ``(N, 3, H, W)``.
            boxes (torch.Tensor): Bounding box coordinates of shape ``(M, 5)``.
                First column indicates batch index of the bbox, remaining columns
                are coordinates ``[x1, y1, x2, y2]``.

        Returns:
            list[torch.Tensor]: List of pose feature tensors for each image, where
                each tensor contains normalized keypoint coordinates.
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
