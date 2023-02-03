"""Region Based Anomaly Detection With Real-Time Training and Analysis."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from anomalib.models.components import AnomalyModule
from anomalib.models.components.classification import FeatureScalingMethod

from .region_extractor import RoiStage
from .torch_model import RkdeModel

logger = logging.getLogger(__name__)


class Rkde(AnomalyModule):
    """Region Based Anomaly Detection With Real-Time Training and Analysis.

    Args:
        roi_stage (RoiStage, optional): Processing stage from which rois are extracted.
        roi_score_threshold (float, optional): Mimumum confidence score for the region proposals.
        min_size (int, optional): Minimum size in pixels for the region proposals.
        iou_threshold (float, optional): Intersection-Over-Union threshold used during NMS.
        max_detections_per_image (int, optional): Maximum number of region proposals per image.
        n_pca_components (int, optional): Number of PCA components. Defaults to 16.
        feature_scaling_method (FeatureScalingMethod, optional): Scaling method applied to features before passing to
            KDE. Options are `norm` (normalize to unit vector length) and `scale` (scale to max length observed in
            training).
        max_training_points (int, optional): Maximum number of training points to fit the KDE model. Defaults to 40000.
    """

    def __init__(
        self,
        roi_stage: RoiStage = RoiStage.RCNN,
        roi_score_threshold: float = 0.001,
        min_box_size: int = 25,
        iou_threshold: float = 0.3,
        max_detections_per_image: int = 100,
        n_pca_components: int = 16,
        feature_scaling_method: FeatureScalingMethod = FeatureScalingMethod.SCALE,
        max_training_points: int = 40000,
    ) -> None:
        super().__init__()

        self.model: RkdeModel = RkdeModel(
            roi_stage=roi_stage,
            roi_score_threshold=roi_score_threshold,
            min_box_size=min_box_size,
            iou_threshold=iou_threshold,
            max_detections_per_image=max_detections_per_image,
            n_pca_components=n_pca_components,
            feature_scaling_method=feature_scaling_method,
            max_training_points=max_training_points,
        )
        self.embeddings: list[Tensor] = []

    @staticmethod
    def configure_optimizers() -> None:
        """RKDE doesn't require optimization, therefore returns no optimizers."""
        return None

    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> None:
        """Training Step of RKDE. For each batch, features are extracted from the CNN.

        Args:
            batch (dict[str, str | Tensor]): Batch containing image filename, image, label and mask

        Returns:
          Deep CNN features.
        """
        del args, kwargs  # These variables are not used.

        features = self.model(batch["image"])
        self.embeddings.append(features)

    def on_validation_start(self) -> None:
        """Fit a KDE Model to the embedding collected from the training set."""
        embeddings = torch.vstack(self.embeddings)

        logger.info("Fitting a KDE model to the embedding collected from the training set.")
        self.model.fit(embeddings)

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Validation Step of RKde.

        Similar to the training step, features are extracted from the CNN for each batch.

        Args:
            batch (dict[str, str | Tensor]): Batch containing image filename, image, label and mask

        Returns:
          Dictionary containing probability, prediction and ground truth values.
        """
        del args, kwargs  # These variables are not used.

        # get batched model predictions
        boxes, scores = self.model(batch["image"])

        # convert batched predictions to list format
        image: Tensor = batch["image"]
        batch_size = image.shape[0]
        indices = boxes[:, 0]
        batch["pred_boxes"] = [boxes[indices == i, 1:] for i in range(batch_size)]
        batch["box_scores"] = [scores[indices == i] for i in range(batch_size)]

        return batch


class RkdeLightning(Rkde):
    """Rkde: Deep Feature Kernel Density Estimation.

    Args:
        hparams (DictConfig | ListConfig): Model params
    """

    def __init__(self, hparams: DictConfig | ListConfig) -> None:
        super().__init__(
            roi_stage=RoiStage(hparams.model.roi_stage),
            roi_score_threshold=hparams.model.roi_score_threshold,
            min_box_size=hparams.model.min_box_size,
            iou_threshold=hparams.model.iou_threshold,
            max_detections_per_image=hparams.model.max_detections_per_image,
            n_pca_components=hparams.model.n_pca_components,
            feature_scaling_method=FeatureScalingMethod(hparams.model.feature_scaling_method),
            max_training_points=hparams.model.max_training_points,
        )
        self.hparams: DictConfig | ListConfig  # type: ignore
        self.save_hyperparameters(hparams)
