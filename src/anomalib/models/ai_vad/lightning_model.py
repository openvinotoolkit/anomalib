"""Attribute-based Representations for Accurate and Interpretable Video Anomaly Detection.

Paper https://arxiv.org/pdf/2212.00789.pdf
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from typing import List
import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from anomalib.models.components import AnomalyModule
from anomalib.models.ai_vad.torch_model import AiVadModel

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

    def __init__(self) -> None:
        super().__init__()

        self.model = AiVadModel()

        self.velocity_embeddings: List[Tensor] = []
        self.pose_embeddings: List[Tensor] = []
        self.feature_embeddings: List[Tensor] = []

        self.pose_embeddings_dict: dict = {}
        self.feature_embeddings_dict: dict = {}

    @staticmethod
    def configure_optimizers() -> None:  # pylint: disable=arguments-differ
        return None

    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> None:
        velocity, poses, features = self.model(batch["image"])
        # add velocity

        # # add poses and features to membanks
        for velocity_embeddings, pose_embeddings, feature_embeddings, video_path in zip(
            velocity, poses, features, batch["video_path"]
        ):
            self.velocity_embeddings.append(velocity_embeddings.cpu())
            self.pose_embeddings.append(pose_embeddings.cpu())
            self.feature_embeddings.append(feature_embeddings.cpu())

            if video_path in self.pose_embeddings_dict:
                self.pose_embeddings_dict[video_path].append(pose_embeddings.cpu())
            else:
                self.pose_embeddings_dict[video_path] = [pose_embeddings.cpu()]

            if video_path in self.feature_embeddings_dict:
                self.feature_embeddings_dict[video_path].append(feature_embeddings.cpu())
            else:
                self.feature_embeddings_dict[video_path] = [feature_embeddings.cpu()]

    def on_validation_start(self) -> None:
        # stack velocity embeddings
        velocity_embeddings = torch.vstack(self.velocity_embeddings)
        # pass to torch model
        self.model.velocity_estimator.fit(velocity_embeddings)
        self.model.velocity_embeddings = velocity_embeddings
        # stack pose embeddings
        self.model.pose_embeddings = [torch.vstack(embeddings) for embeddings in self.pose_embeddings_dict.values()]
        # stack feature embeddings
        self.model.feature_embeddings = [
            torch.vstack(embeddings) for embeddings in self.feature_embeddings_dict.values()
        ]

        self.model.compute_normalization_statistics()

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        boxes, anomaly_scores = self.model(batch["image"])
        batch["pred_boxes"] = [box.int() for box in boxes]
        batch["box_scores"] = [score.to(self.device) for score in anomaly_scores]

        # TODO: this should be handled by video dataset
        batch["boxes"] = [boxes[-1] for boxes in batch["boxes"]]
        batch["mask"] = batch["mask"][:, -1, ...]
        batch["image"] = batch["image"][:, -1, ...]
        batch["original_image"] = batch["original_image"][:, -1, ...]
        batch["label"] = batch["label"][:, -1]
        batch["frames"] = batch["frames"][:, -1]

        return batch


class AiVadLightning(AiVad):
    """PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization.

    Args:
        hparams (DictConfig | ListConfig): Model params
    """

    def __init__(self, hparams: DictConfig | ListConfig) -> None:
        super().__init__()
        self.hparams: DictConfig | ListConfig  # type: ignore
        self.save_hyperparameters(hparams)
