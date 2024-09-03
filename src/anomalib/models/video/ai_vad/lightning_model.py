"""Attribute-based Representations for Accurate and Interpretable Video Anomaly Detection.

Paper https://arxiv.org/pdf/2212.00789.pdf
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.transforms.v2 import Transform

from anomalib import LearningType
from anomalib.models.components import AnomalyModule, MemoryBankMixin

from .torch_model import AiVadModel

logger = logging.getLogger(__name__)

__all__ = ["AiVad"]


class AiVad(MemoryBankMixin, AnomalyModule):
    """AI-VAD: Attribute-based Representations for Accurate and Interpretable Video Anomaly Detection.

    Args:
        box_score_thresh (float): Confidence threshold for bounding box predictions.
            Defaults to ``0.7``.
        persons_only (bool): When enabled, only regions labeled as person are included.
            Defaults to ``False``.
        min_bbox_area (int): Minimum bounding box area. Regions with a surface area lower than this value are excluded.
            Defaults to ``100``.
        max_bbox_overlap (float): Maximum allowed overlap between bounding boxes.
            Defaults to ``0.65``.
        enable_foreground_detections (bool): Add additional foreground detections based on pixel difference between
            consecutive frames.
            Defaults to ``True``.
        foreground_kernel_size (int): Gaussian kernel size used in foreground detection.
            Defaults to ``3``.
        foreground_binary_threshold (int): Value between 0 and 255 which acts as binary threshold in foreground
            detection.
            Defaults to ``18``.
        n_velocity_bins (int): Number of discrete bins used for velocity histogram features.
            Defaults to ``1``.
        use_velocity_features (bool): Flag indicating if velocity features should be used.
            Defaults to ``True``.
        use_pose_features (bool): Flag indicating if pose features should be used.
            Defaults to ``True``.
        use_deep_features (bool): Flag indicating if deep features should be used.
            Defaults to ``True``.
        n_components_velocity (int): Number of components used by GMM density estimation for velocity features.
            Defaults to ``2``.
        n_neighbors_pose (int): Number of neighbors used in KNN density estimation for pose features.
            Defaults to ``1``.
        n_neighbors_deep (int): Number of neighbors used in KNN density estimation for deep features.
            Defaults to ``1``.
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
    ) -> None:
        super().__init__()

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

    def training_step(self, batch: dict[str, str | torch.Tensor]) -> None:
        """Training Step of AI-VAD.

        Extract features from the batch of clips and update the density estimators.

        Args:
            batch (dict[str, str | torch.Tensor]): Batch containing image filename, image, label and mask
        """
        features_per_batch = self.model(batch["image"])

        for features, video_path in zip(features_per_batch, batch["video_path"], strict=True):
            self.model.density_estimator.update(features, video_path)
            self.total_detections += len(next(iter(features.values())))

    def fit(self) -> None:
        """Fit the density estimators to the extracted features from the training set."""
        if self.total_detections == 0:
            msg = "No regions were extracted during training."
            raise ValueError(msg)
        self.model.density_estimator.fit()

    def validation_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Perform the validation step of AI-VAD.

        Extract boxes and box scores..

        Args:
            batch (dict[str, str | torch.Tensor]): Input batch
            *args: Arguments.
            **kwargs: Keyword arguments.

        Returns:
            Batch dictionary with added boxes and box scores.
        """
        del args, kwargs  # Unused arguments.

        boxes, anomaly_scores, image_scores = self.model(batch["image"])
        batch["pred_boxes"] = [box.int() for box in boxes]
        batch["box_scores"] = [score.to(self.device) for score in anomaly_scores]
        batch["pred_scores"] = torch.Tensor(image_scores).to(self.device)

        return batch

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """AI-VAD specific trainer arguments."""
        return {"gradient_clip_val": 0, "max_epochs": 1, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS

    @staticmethod
    def configure_transforms(image_size: tuple[int, int] | None = None) -> Transform | None:
        """AI-VAD does not need a transform, as the region- and feature-extractors apply their own transforms."""
        del image_size
        return None
