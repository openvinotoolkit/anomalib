"""AI-VAD.

Attribute-based Representations for Accurate and Interpretable Video Anomaly
Detection.

This module implements the AI-VAD model as described in the paper "AI-VAD:
Attribute-based Representations for Accurate and Interpretable Video Anomaly
Detection."

The model extracts regions of interest from video frames using object detection and
foreground detection, then computes attribute-based representations including
velocity, pose and deep features for anomaly detection.

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
    Tal Reiss, Yedid Hoshen. "AI-VAD: Attribute-based Representations for Accurate
    and Interpretable Video Anomaly Detection." arXiv preprint arXiv:2212.00789
    (2022). https://arxiv.org/pdf/2212.00789.pdf
"""

# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn

from anomalib import LearningType
from anomalib.data import VideoBatch
from anomalib.models.components import AnomalibModule, MemoryBankMixin
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor

from .torch_model import AiVadModel

logger = logging.getLogger(__name__)

__all__ = ["AiVad"]


class AiVad(MemoryBankMixin, AnomalibModule):
    """AI-VAD: Attribute-based Representations for Video Anomaly Detection.

    This model extracts regions of interest from video frames using object detection and
    foreground detection, then computes attribute-based representations including
    velocity, pose and deep features for anomaly detection.

    Args:
        box_score_thresh (float, optional): Confidence threshold for bounding box
            predictions. Defaults to ``0.7``.
        persons_only (bool, optional): When enabled, only regions labeled as person are
            included. Defaults to ``False``.
        min_bbox_area (int, optional): Minimum bounding box area. Regions with surface
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
            histogram features. Defaults to ``1``.
        use_velocity_features (bool, optional): Flag indicating if velocity features
            should be used. Defaults to ``True``.
        use_pose_features (bool, optional): Flag indicating if pose features should be
            used. Defaults to ``True``.
        use_deep_features (bool, optional): Flag indicating if deep features should be
            used. Defaults to ``True``.
        n_components_velocity (int, optional): Number of components used by GMM density
            estimation for velocity features. Defaults to ``2``.
        n_neighbors_pose (int, optional): Number of neighbors used in KNN density
            estimation for pose features. Defaults to ``1``.
        n_neighbors_deep (int, optional): Number of neighbors used in KNN density
            estimation for deep features. Defaults to ``1``.
        pre_processor (PreProcessor | bool, optional): Pre-processor instance or bool
            flag to enable default pre-processor. Defaults to ``True``.
        post_processor (PostProcessor | bool, optional): Post-processor instance or bool
            flag to enable default post-processor. Defaults to ``True``.
        **kwargs: Additional keyword arguments passed to the parent class.

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

    Note:
        The model follows a one-class learning approach and does not require
        optimization during training. Instead, it builds density estimators based on
        extracted features from normal samples.
    """

    def __init__(
        self,
        box_score_thresh: float = 0.7,
        persons_only: bool = False,
        min_bbox_area: int = 100,
        max_bbox_overlap: float = 0.65,
        enable_foreground_detections: bool = True,
        foreground_kernel_size: int = 3,
        foreground_binary_threshold: int = 18,
        n_velocity_bins: int = 1,
        use_velocity_features: bool = True,
        use_pose_features: bool = True,
        use_deep_features: bool = True,
        n_components_velocity: int = 2,
        n_neighbors_pose: int = 1,
        n_neighbors_deep: int = 1,
        pre_processor: nn.Module | bool = True,
        post_processor: nn.Module | bool = True,
        **kwargs,
    ) -> None:
        super().__init__(pre_processor=pre_processor, post_processor=post_processor, **kwargs)
        self.model = AiVadModel(
            box_score_thresh=box_score_thresh,
            persons_only=persons_only,
            min_bbox_area=min_bbox_area,
            max_bbox_overlap=max_bbox_overlap,
            enable_foreground_detections=enable_foreground_detections,
            foreground_kernel_size=foreground_kernel_size,
            foreground_binary_threshold=foreground_binary_threshold,
            n_velocity_bins=n_velocity_bins,
            use_velocity_features=use_velocity_features,
            use_pose_features=use_pose_features,
            use_deep_features=use_deep_features,
            n_components_velocity=n_components_velocity,
            n_neighbors_pose=n_neighbors_pose,
            n_neighbors_deep=n_neighbors_deep,
        )

        self.total_detections = 0

    @staticmethod
    def configure_optimizers() -> None:
        """AI-VAD training does not involve fine-tuning of NN weights, no optimizers needed."""
        return

    def training_step(self, batch: VideoBatch) -> None:
        """Training Step of AI-VAD.

        Extract features from the batch of clips and update the density estimators.

        Args:
            batch (VideoBatch): Batch containing video frames and metadata.
        """
        features_per_batch = self.model(batch.image)

        assert isinstance(batch.video_path, list)
        for features, video_path in zip(features_per_batch, batch.video_path, strict=True):
            self.model.density_estimator.update(features, video_path)
            self.total_detections += len(next(iter(features.values())))

        # Return a dummy loss tensor
        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def fit(self) -> None:
        """Fit the density estimators to the extracted features from the training set.

        Raises:
            ValueError: If no regions were extracted during training.
        """
        if self.total_detections == 0:
            msg = "No regions were extracted during training."
            raise ValueError(msg)
        self.model.density_estimator.fit()

    def validation_step(self, batch: VideoBatch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform the validation step of AI-VAD.

        Extract boxes and box scores from the input batch.

        Args:
            batch (VideoBatch): Input batch containing video frames and metadata.
            *args: Additional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            STEP_OUTPUT: Batch dictionary with added predictions and anomaly maps.
        """
        del args, kwargs  # Unused arguments.

        predictions = self.model(batch.image)
        return batch.update(pred_score=predictions.pred_score, anomaly_map=predictions.anomaly_map)

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Get AI-VAD specific trainer arguments.

        Returns:
            dict[str, Any]: Dictionary of trainer arguments.
        """
        return {"gradient_clip_val": 0, "max_epochs": 1, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Get the learning type of the model.

        Returns:
            LearningType: Learning type of the model (ONE_CLASS).
        """
        return LearningType.ONE_CLASS

    @classmethod
    def configure_pre_processor(cls, image_size: tuple[int, int] | None = None) -> PreProcessor:
        """Configure the pre-processor for AI-VAD.

        AI-VAD does not need a pre-processor or transforms, as the region- and
        feature-extractors apply their own transforms.

        Args:
            image_size (tuple[int, int] | None, optional): Image size (unused).
                Defaults to ``None``.

        Returns:
            PreProcessor: Empty pre-processor instance.
        """
        del image_size
        return PreProcessor()  # A pre-processor with no transforms.

    @staticmethod
    def configure_post_processor() -> PostProcessor:
        """Configure the post-processor for AI-VAD.

        Returns:
            PostProcessor: One-class post-processor instance.
        """
        return PostProcessor()
