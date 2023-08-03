"""Attribute-based Representations for Accurate and Interpretable Video Anomaly Detection.

Paper https://arxiv.org/pdf/2212.00789.pdf
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from anomalib.models.ai_vad.torch_model import AiVadModel
from anomalib.models.components import AnomalyModule

logger = logging.getLogger(__name__)

__all__ = ["AiVad", "AiVadLightning"]


class AiVad(AnomalyModule):
    """AI-VAD: Attribute-based Representations for Accurate and Interpretable Video Anomaly Detection.

    Args:
        box_score_thresh (float): Confidence threshold for bounding box predictions.
        persons_only (bool): When enabled, only regions labeled as person are included.
        min_bbox_area (int): Minimum bounding box area. Regions with a surface area lower than this value are excluded.
        max_bbox_overlap (float): Maximum allowed overlap between bounding boxes.
        enable_foreground_detections (bool): Add additional foreground detections based on pixel difference between
            consecutive frames.
        foreground_kernel_size (int): Gaussian kernel size used in foreground detection.
        foreground_binary_threshold (int): Value between 0 and 255 which acts as binary threshold in foreground
            detection.
        n_velocity_bins (int): Number of discrete bins used for velocity histogram features.
        use_velocity_features (bool): Flag indicating if velocity features should be used.
        use_pose_features (bool): Flag indicating if pose features should be used.
        use_deep_features (bool): Flag indicating if deep features should be used.
        n_components_velocity (int): Number of components used by GMM density estimation for velocity features.
        n_neighbors_pose (int): Number of neighbors used in KNN density estimation for pose features.
        n_neighbors_deep (int): Number of neighbors used in KNN density estimation for deep features.
    """

    def __init__(
        self,
        box_score_thresh: float = 0.8,
        persons_only: bool = False,
        min_bbox_area: int = 100,
        max_bbox_overlap: float = 0.65,
        enable_foreground_detections: bool = True,
        foreground_kernel_size: int = 3,
        foreground_binary_threshold: int = 18,
        n_velocity_bins: int = 8,
        use_velocity_features: bool = True,
        use_pose_features: bool = True,
        use_deep_features: bool = True,
        n_components_velocity: int = 5,
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

    @staticmethod
    def configure_optimizers() -> None:
        """AI-VAD training does not involve fine-tuning of NN weights, no optimizers needed."""
        return None

    def training_step(self, batch: dict[str, str | Tensor]) -> None:
        """Training Step of AI-VAD.

        Extract features from the batch of clips and update the density estimators.

        Args:
            batch (dict[str, str | Tensor]): Batch containing image filename, image, label and mask
        """
        features_per_batch = self.model(batch["image"])

        for features, video_path in zip(features_per_batch, batch["video_path"]):
            self.model.density_estimator.update(features, video_path)

    def on_validation_start(self) -> None:
        """Fit the density estimators to the extracted features from the training set."""
        # NOTE: Previous anomalib versions fit Gaussian at the end of the epoch.
        #   This is not possible anymore with PyTorch Lightning v1.4.0 since validation
        #   is run within train epoch.
        self.model.density_estimator.fit()

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Validation Step of AI-VAD.

        Extract boxes and box scores..

        Args:
            batch (dict[str, str | Tensor]): Input batch

        Returns:
            Batch dictionary with added boxes and box scores.
        """
        boxes, anomaly_scores, image_scores = self.model(batch["image"])
        batch["pred_boxes"] = [box.int() for box in boxes]
        batch["box_scores"] = [score.to(self.device) for score in anomaly_scores]
        batch["pred_scores"] = Tensor(image_scores).to(self.device)

        return batch


class AiVadLightning(AiVad):
    """AI-VAD: Attribute-based Representations for Accurate and Interpretable Video Anomaly Detection.

    Args:
        hparams (DictConfig | ListConfig): Model params
    """

    def __init__(self, hparams: DictConfig | ListConfig) -> None:
        super().__init__(
            box_score_thresh=hparams.model.box_score_thresh,
            persons_only=hparams.model.persons_only,
            min_bbox_area=hparams.model.min_bbox_area,
            max_bbox_overlap=hparams.model.max_bbox_overlap,
            enable_foreground_detections=hparams.model.enable_foreground_detections,
            foreground_kernel_size=hparams.model.foreground_kernel_size,
            foreground_binary_threshold=hparams.model.foreground_binary_threshold,
            n_velocity_bins=hparams.model.n_velocity_bins,
            use_velocity_features=hparams.model.use_velocity_features,
            use_pose_features=hparams.model.use_pose_features,
            use_deep_features=hparams.model.use_deep_features,
            n_components_velocity=hparams.model.n_components_velocity,
            n_neighbors_pose=hparams.model.n_neighbors_pose,
            n_neighbors_deep=hparams.model.n_neighbors_deep,
        )
        self.hparams: DictConfig | ListConfig  # type: ignore
        self.save_hyperparameters(hparams)
