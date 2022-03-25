"""DFM: Deep Feature Kernel Density Estimation."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import List, Union

import torch
from omegaconf import DictConfig, ListConfig
from torch import Tensor

from anomalib.models.components import AnomalyModule

from .dfm_model import DFMModel


class DfmLightning(AnomalyModule):
    """DFM: Deep Featured Kernel Density Estimation."""

    def __init__(self, hparams: Union[DictConfig, ListConfig]):
        super().__init__(hparams)

        self.model: DFMModel = DFMModel(
            backbone=hparams.model.backbone, n_comps=hparams.model.pca_level, score_type=hparams.model.score_type
        )
        self.automatic_optimization = False
        self.embeddings: List[Tensor] = []

    @staticmethod
    def configure_optimizers() -> None:  # pylint: disable=arguments-differ
        """DFM doesn't require optimization, therefore returns no optimizers."""
        return None

    def training_step(self, batch, _):  # pylint: disable=arguments-differ
        """Training Step of DFM.

        For each batch, features are extracted from the CNN.

        Args:
          batch (Dict[str, Tensor]): Input batch
          _: Index of the batch.

        Returns:
          Deep CNN features.
        """
        embedding = self.model.get_features(batch["image"]).squeeze()

        # NOTE: `self.embedding` appends each batch embedding to
        #   store the training set embedding. We manually append these
        #   values mainly due to the new order of hooks introduced after PL v1.4.0
        #   https://github.com/PyTorchLightning/pytorch-lightning/pull/7357
        self.embeddings.append(embedding)

    def on_validation_start(self) -> None:
        """Fit a KDE Model to the embedding collected from the training set."""
        # NOTE: Previous anomalib versions fit Gaussian at the end of the epoch.
        #   This is not possible anymore with PyTorch Lightning v1.4.0 since validation
        #   is run within train epoch.
        embeddings = torch.vstack(self.embeddings)
        self.model.fit(embeddings)

    def validation_step(self, batch, _):  # pylint: disable=arguments-differ
        """Validation Step of DFM.

        Similar to the training step, features are extracted from the CNN for each batch.

        Args:
          batch (List[Dict[str, Any]]): Input batch

        Returns:
          Dictionary containing FRE anomaly scores and ground-truth.
        """
        batch["pred_scores"] = self.model(batch["image"])

        return batch
