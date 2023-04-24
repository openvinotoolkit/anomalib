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
    """PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization.

    Args:
        layers (list[str]): Layers to extract features from the backbone CNN
        input_size (tuple[int, int]): Size of the model input.
        backbone (str): Backbone CNN network
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
        n_features (int, optional): Number of features to retain in the dimension reduction step.
                                Default values from the paper are available for: resnet18 (100), wide_resnet50_2 (550).
    """

    def __init__(
        self,
        box_score_thresh: float = 0.8,
        n_velocity_bins: int = 8,
        use_velocity_features: bool = True,
        use_pose_features: bool = True,
        use_appearance_features: bool = True,
        n_components_velocity: int = 5,
        n_neighbors_pose: int = 1,
        n_neighbors_appearance: int = 1,
    ) -> None:
        super().__init__()

        self.model = AiVadModel(
            box_score_thresh=box_score_thresh,
            n_velocity_bins=n_velocity_bins,
            use_velocity_features=use_velocity_features,
            use_pose_features=use_pose_features,
            use_appearance_features=use_appearance_features,
            n_components_velocity=n_components_velocity,
            n_neighbors_pose=n_neighbors_pose,
            n_neighbors_appearance=n_neighbors_appearance,
        )

    @staticmethod
    def configure_optimizers() -> None:  # pylint: disable=arguments-differ
        """TAI-VAD training does not involve fine-tuning of NN weights, no optimizers needed."""
        return None

    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> None:
        """Training Step of AI-VAD.

        Extract features from the batch of clips and update the density estimators.

        Args:
            batch (dict[str, str | Tensor]): Batch containing image filename, image, label and mask
        """
        features_per_batch = self.model(batch)

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
        boxes, anomaly_scores = self.model(batch)
        batch["pred_boxes"] = [box.int() for box in boxes]
        batch["box_scores"] = [score.to(self.device) for score in anomaly_scores]

        return batch


class AiVadLightning(AiVad):
    """PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization.

    Args:
        hparams (DictConfig | ListConfig): Model params
    """

    def __init__(self, hparams: DictConfig | ListConfig) -> None:
        super().__init__(
            box_score_thresh=hparams.model.box_score_thresh,
            n_velocity_bins=hparams.model.n_velocity_bins,
            use_velocity_features=hparams.model.use_velocity_features,
            use_pose_features=hparams.model.use_pose_features,
            use_appearance_features=hparams.model.use_appearance_features,
            n_components_velocity=hparams.model.n_components_velocity,
            n_neighbors_pose=hparams.model.n_neighbors_pose,
            n_neighbors_appearance=hparams.model.n_neighbors_appearance,
        )
        self.hparams: DictConfig | ListConfig  # type: ignore
        self.save_hyperparameters(hparams)
