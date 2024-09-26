"""Region Based Anomaly Detection With Real-Time Training and Analysis.

https://ieeexplore.ieee.org/abstract/document/8999287
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.transforms.v2 import Compose, Resize, Transform

from anomalib import LearningType
from anomalib.models.components import AnomalyModule, MemoryBankMixin
from anomalib.models.components.classification import FeatureScalingMethod

from .region_extractor import RoiStage
from .torch_model import RkdeModel

logger = logging.getLogger(__name__)


class Rkde(MemoryBankMixin, AnomalyModule):
    """Region Based Anomaly Detection With Real-Time Training and Analysis.

    Args:
        roi_stage (RoiStage, optional): Processing stage from which rois are extracted.
            Defaults to ``RoiStage.RCNN``.
        roi_score_threshold (float, optional): Mimumum confidence score for the region proposals.
            Defaults to ``0.001``.
        min_size (int, optional): Minimum size in pixels for the region proposals.
            Defaults to ``25``.
        iou_threshold (float, optional): Intersection-Over-Union threshold used during NMS.
            Defaults to ``0.3``.
        max_detections_per_image (int, optional): Maximum number of region proposals per image.
            Defaults to ``100``.
        n_pca_components (int, optional): Number of PCA components.
            Defaults to ``16``.
        feature_scaling_method (FeatureScalingMethod, optional): Scaling method applied to features before passing to
            KDE. Options are `norm` (normalize to unit vector length) and `scale` (scale to max length observed in
            training).
            Defaults to ``FeatureScalingMethod.SCALE``.
        max_training_points (int, optional): Maximum number of training points to fit the KDE model.
            Defaults to ``40000``.
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
        self.embeddings: list[torch.Tensor] = []

    @staticmethod
    def configure_optimizers() -> None:
        """RKDE doesn't require optimization, therefore returns no optimizers."""
        return

    def training_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> None:
        """Perform a training Step of RKDE. For each batch, features are extracted from the CNN.

        Args:
            batch (dict[str, str | torch.Tensor]): Batch containing image filename, image, label and mask
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
          Deep CNN features.
        """
        del args, kwargs  # These variables are not used.

        features = self.model(batch["image"])
        self.embeddings.append(features)

    def fit(self) -> None:
        """Fit a KDE Model to the embedding collected from the training set."""
        embeddings = torch.vstack(self.embeddings)

        logger.info("Fitting a KDE model to the embedding collected from the training set.")
        self.model.fit(embeddings)

    def validation_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Perform a validation Step of RKde.

        Similar to the training step, features are extracted from the CNN for each batch.

        Args:
            batch (dict[str, str | torch.Tensor]): Batch containing image filename, image, label and mask
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
          Dictionary containing probability, prediction and ground truth values.
        """
        del args, kwargs  # These variables are not used.

        # get batched model predictions
        boxes, scores = self.model(batch["image"])

        # convert batched predictions to list format
        image: torch.Tensor = batch["image"]
        batch_size = image.shape[0]
        indices = boxes[:, 0]
        batch["pred_boxes"] = [boxes[indices == i, 1:] for i in range(batch_size)]
        batch["box_scores"] = [scores[indices == i] for i in range(batch_size)]

        return batch

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return R-KDE trainer arguments.

        Returns:
            dict[str, Any]: Arguments for the trainer.
        """
        return {"gradient_clip_val": 0, "max_epochs": 1, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS

    @staticmethod
    def configure_transforms(image_size: tuple[int, int] | None = None) -> Transform:
        """Default transform for RKDE."""
        image_size = image_size or (240, 360)
        return Compose(
            [
                Resize(image_size, antialias=True),
            ],
        )
