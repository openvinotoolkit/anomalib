"""DFKDE: Deep Feature Kernel Density Estimation."""

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
import torchvision
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch import Tensor, nn

from anomalib.models.components import AnomalyModule, FeatureExtractor

from .normality_model import NormalityModel


class DfkdeModel(nn.Module):
    """DFKDE model.

    Args:
        backbone (str): Pre-trained model backbone.
        filter_count (int): Number of filters.
        threshold_steepness (float): Threshold steepness for normality model.
        threshold_offset (float): Threshold offset for normality model.
    """

    def __init__(self, backbone: str, filter_count: int, threshold_steepness: float, threshold_offset: float) -> None:
        super().__init__()
        self.backbone = getattr(torchvision.models, backbone)
        self.feature_extractor = FeatureExtractor(backbone=self.backbone(pretrained=True), layers=["avgpool"]).eval()

        self.normality_model = NormalityModel(
            filter_count=filter_count,
            threshold_steepness=threshold_steepness,
            threshold_offset=threshold_offset,
        )

    def get_features(self, batch: Tensor) -> Tensor:
        """Extract features from the pretrained network.

        Args:
            batch (Tensor): Image batch.

        Returns:
            Tensor: Tensor containing extracted features.
        """
        self.feature_extractor.eval()
        layer_outputs = self.feature_extractor(batch)
        layer_outputs = torch.cat(list(layer_outputs.values())).detach()
        return layer_outputs

    def fit(self, embeddings: List[Tensor]):
        """Fit normality model.

        Args:
            embeddings (List[Tensor]): Embeddings to fit.
        """
        _embeddings = torch.vstack(embeddings)
        self.normality_model.fit(_embeddings)

    def forward(self, batch: Tensor) -> Tensor:
        """Prediction by normality model.

        Args:
            batch (Tensor): Input images.

        Returns:
            Tensor: Predictions
        """
        feature_vector = self.get_features(batch)
        return self.normality_model.predict(feature_vector.view(feature_vector.shape[:2]))


class DfkdeLightning(AnomalyModule):
    """DFKDE: Deep Feature Kernel Density Estimation.

    Args:
        hparams (Union[DictConfig, ListConfig]): Model params
    """

    def __init__(self, hparams: Union[DictConfig, ListConfig]):
        super().__init__(hparams)
        threshold_steepness = 0.05
        threshold_offset = 12

        self.model: DfkdeModel = DfkdeModel(
            hparams.model.backbone, hparams.model.max_training_points, threshold_steepness, threshold_offset
        )

        self.automatic_optimization = False
        self.embeddings: List[Tensor] = []

    @staticmethod
    def configure_optimizers():  # pylint: disable=arguments-differ
        """DFKDE doesn't require optimization, therefore returns no optimizers."""
        return None

    def training_step(self, batch, _batch_idx):  # pylint: disable=arguments-differ
        """Training Step of DFKDE. For each batch, features are extracted from the CNN.

        Args:
            batch (Dict[str, Any]): Batch containing image filename, image, label and mask
            _batch_idx: Index of the batch.

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
        self.model.fit(self.embeddings)

    def validation_step(self, batch, _):  # pylint: disable=arguments-differ
        """Validation Step of DFKDE.

        Similar to the training step, features are extracted from the CNN for each batch.

        Args:
          batch: Input batch

        Returns:
          Dictionary containing probability, prediction and ground truth values.
        """
        batch["pred_scores"] = self.model(batch["image"])

        return batch
