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

        self.pose_embeddings: List[Tensor] = []
        self.feature_embeddings: List[Tensor] = []

    @staticmethod
    def configure_optimizers() -> None:  # pylint: disable=arguments-differ
        return None

    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> None:
        velocity, poses, features = self.model(batch["image"])
        # add velocity

        # # add poses and features to membanks
        for pose_embeddings, feature_embeddings in zip(poses, features):
            self.pose_embeddings.append(pose_embeddings.cpu())
            self.feature_embeddings.append(feature_embeddings.cpu())

    def on_validation_start(self) -> None:
        # stack pose embeddings
        pose_embeddings = torch.vstack(self.pose_embeddings)
        # pass to torch model
        self.model.pose_embeddings = pose_embeddings
        # stack feature embeddings
        feature_embeddings = torch.vstack(self.feature_embeddings)
        # pass to torch model
        self.model.feature_embeddings = feature_embeddings

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
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
