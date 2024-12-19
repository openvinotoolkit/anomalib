"""PyTorch model for AI-VAD model implementation.

This module implements the AI-VAD model as described in the paper
"AI-VAD: Attribute-based Representations for Accurate and Interpretable Video
Anomaly Detection."

Example:
    >>> from anomalib.models.video import AiVad
    >>> from anomalib.data import Avenue
    >>> from anomalib.data.utils import VideoTargetFrame
    >>> from anomalib.engine import Engine

    >>> # Initialize model and datamodule
    >>> datamodule = Avenue(
    ...     clip_length_in_frames=2,
    ...     frames_between_clips=1,
    ...     target_frame=VideoTargetFrame.LAST
    ... )
    >>> model = AiVad()

    >>> # Train using the engine
    >>> engine = Engine()
    >>> engine.fit(model=model, datamodule=datamodule)

Reference:
    Tal Reiss, Yedid Hoshen. "AI-VAD: Attribute-based Representations for Accurate and
    Interpretable Video Anomaly Detection." arXiv preprint arXiv:2212.00789 (2022).
    https://arxiv.org/pdf/2212.00789.pdf
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from anomalib.data import InferenceBatch

from .density import CombinedDensityEstimator
from .features import FeatureExtractor
from .flow import FlowExtractor
from .regions import RegionExtractor


class AiVadModel(nn.Module):
    """AI-VAD model.

    The model consists of several stages:
    1. Flow extraction between consecutive frames
    2. Region extraction using object detection and foreground detection
    3. Feature extraction including velocity, pose and deep features
    4. Density estimation for anomaly detection

    Args:
        box_score_thresh (float, optional): Confidence threshold for region extraction
            stage. Defaults to ``0.8``.
        persons_only (bool, optional): When enabled, only regions labeled as person are
            included. Defaults to ``False``.
        min_bbox_area (int, optional): Minimum bounding box area. Regions with a surface
            area lower than this value are excluded. Defaults to ``100``.
        max_bbox_overlap (float, optional): Maximum allowed overlap between bounding
            boxes. Defaults to ``0.65``.
        enable_foreground_detections (bool, optional): Add additional foreground
            detections based on pixel difference between consecutive frames.
            Defaults to ``True``.
        foreground_kernel_size (int, optional): Gaussian kernel size used in foreground
            detection. Defaults to ``3``.
        foreground_binary_threshold (int, optional): Value between 0 and 255 which acts
            as binary threshold in foreground detection. Defaults to ``18``.
        n_velocity_bins (int, optional): Number of discrete bins used for velocity
            histogram features. Defaults to ``8``.
        use_velocity_features (bool, optional): Flag indicating if velocity features
            should be used. Defaults to ``True``.
        use_pose_features (bool, optional): Flag indicating if pose features should be
            used. Defaults to ``True``.
        use_deep_features (bool, optional): Flag indicating if deep features should be
            used. Defaults to ``True``.
        n_components_velocity (int, optional): Number of components used by GMM density
            estimation for velocity features. Defaults to ``5``.
        n_neighbors_pose (int, optional): Number of neighbors used in KNN density
            estimation for pose features. Defaults to ``1``.
        n_neighbors_deep (int, optional): Number of neighbors used in KNN density
            estimation for deep features. Defaults to ``1``.

    Raises:
        ValueError: If none of the feature types (velocity, pose, deep) are enabled.

    Example:
        >>> from anomalib.models.video.ai_vad.torch_model import AiVadModel
        >>> model = AiVadModel()
        >>> batch = torch.randn(32, 2, 3, 256, 256)  # (N, L, C, H, W)
        >>> output = model(batch)
        >>> output.pred_score.shape
        torch.Size([32])
        >>> output.anomaly_map.shape
        torch.Size([32, 256, 256])
    """

    def __init__(
        self,
        # region-extraction params
        box_score_thresh: float = 0.8,
        persons_only: bool = False,
        min_bbox_area: int = 100,
        max_bbox_overlap: float = 0.65,
        enable_foreground_detections: bool = True,
        foreground_kernel_size: int = 3,
        foreground_binary_threshold: int = 18,
        # feature-extraction params
        n_velocity_bins: int = 8,
        use_velocity_features: bool = True,
        use_pose_features: bool = True,
        use_deep_features: bool = True,
        # density-estimation params
        n_components_velocity: int = 5,
        n_neighbors_pose: int = 1,
        n_neighbors_deep: int = 1,
    ) -> None:
        super().__init__()
        if not any((use_velocity_features, use_pose_features, use_deep_features)):
            msg = "Select at least one feature type."
            raise ValueError(msg)

        # initialize flow extractor
        self.flow_extractor = FlowExtractor()
        # initialize region extractor
        self.region_extractor = RegionExtractor(
            box_score_thresh=box_score_thresh,
            persons_only=persons_only,
            min_bbox_area=min_bbox_area,
            max_bbox_overlap=max_bbox_overlap,
            enable_foreground_detections=enable_foreground_detections,
            foreground_kernel_size=foreground_kernel_size,
            foreground_binary_threshold=foreground_binary_threshold,
        )
        # initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            n_velocity_bins=n_velocity_bins,
            use_velocity_features=use_velocity_features,
            use_pose_features=use_pose_features,
            use_deep_features=use_deep_features,
        )
        # initialize density estimator
        self.density_estimator = CombinedDensityEstimator(
            use_velocity_features=use_velocity_features,
            use_pose_features=use_pose_features,
            use_deep_features=use_deep_features,
            n_components_velocity=n_components_velocity,
            n_neighbors_pose=n_neighbors_pose,
            n_neighbors_deep=n_neighbors_deep,
        )

    def forward(self, batch: torch.Tensor) -> InferenceBatch:
        """Forward pass through AI-VAD model.

        The forward pass consists of the following steps:
        1. Extract first and last frame from input clip
        2. Extract optical flow between frames and detect regions of interest
        3. Extract features (velocity, pose, deep) for each region
        4. Estimate density and compute anomaly scores

        Args:
            batch (torch.Tensor): Input tensor of shape ``(N, L, C, H, W)`` where:
                - ``N``: Batch size
                - ``L``: Sequence length
                - ``C``: Number of channels
                - ``H``: Height
                - ``W``: Width

        Returns:
            InferenceBatch: Batch containing:
                - ``pred_score``: Per-image anomaly scores of shape ``(N,)``
                - ``anomaly_map``: Per-pixel anomaly scores of shape ``(N, H, W)``

        Example:
            >>> batch = torch.randn(32, 2, 3, 256, 256)
            >>> model = AiVadModel()
            >>> output = model(batch)
            >>> output.pred_score.shape, output.anomaly_map.shape
            (torch.Size([32]), torch.Size([32, 256, 256]))
        """
        self.flow_extractor.eval()
        self.region_extractor.eval()
        self.feature_extractor.eval()

        # 1. get first and last frame from clip
        first_frame = batch[:, 0, ...]
        last_frame = batch[:, -1, ...]

        # 2. extract flows and regions
        with torch.no_grad():
            flows = self.flow_extractor(first_frame, last_frame)
            regions = self.region_extractor(first_frame, last_frame)

        # 3. extract pose, appearance and velocity features
        features_per_batch = self.feature_extractor(first_frame, flows, regions)

        if self.training:
            return features_per_batch

        # 4. estimate density
        box_scores = []
        image_scores = []
        for features in features_per_batch:
            box, image = self.density_estimator(features)
            box_scores.append(box)
            image_scores.append(image)

        anomaly_map = torch.stack(
            [
                torch.amax(region["masks"] * scores.view(-1, 1, 1, 1), dim=0)
                for region, scores in zip(regions, box_scores, strict=False)
            ],
        )

        return InferenceBatch(
            pred_score=torch.stack(image_scores),
            anomaly_map=anomaly_map,
        )
