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

from typing import Dict, List, Union

import torch
import torchvision
from omegaconf import DictConfig, ListConfig
from torch import Tensor

from anomalib.core.model import AnomalyModule
from anomalib.core.model.feature_extractor import FeatureExtractor
from anomalib.models.dfm.dfm_model import DFMModel


class DfmLightning(AnomalyModule):
    """DFM: Deep Featured Kernel Density Estimation."""

    def __init__(self, hparams: Union[DictConfig, ListConfig]):
        super().__init__(hparams)

        self.backbone = getattr(torchvision.models, hparams.model.backbone)
        self.feature_extractor = FeatureExtractor(backbone=self.backbone(pretrained=True), layers=["avgpool"]).eval()

        self.dfm_model = DFMModel(n_comps=hparams.model.pca_level, score_type=hparams.model.score_type)
        self.automatic_optimization = False

    @staticmethod
    def configure_optimizers() -> None:
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

        self.feature_extractor.eval()
        layer_outputs = self.feature_extractor(batch["image"])
        feature_vector = torch.hstack(list(layer_outputs.values())).detach().squeeze()
        return {"feature_vector": feature_vector}

    def training_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        """Fit a KDE model on deep CNN features.

        Args:
          outputs (List[Dict[str, Tensor]]): Batch of outputs from the training step

        Returns:
          None
        """

        feature_stack = torch.vstack([output["feature_vector"] for output in outputs])
        self.dfm_model.fit(feature_stack)

    def validation_step(self, batch, _):  # pylint: disable=arguments-differ
        """Validation Step of DFM.

        Similar to the training step, features are extracted from the CNN for each batch.

        Args:
          batch (List[Dict[str, Any]]): Input batch

        Returns:
          Dictionary containing FRE anomaly scores and ground-truth.
        """

        self.feature_extractor.eval()
        layer_outputs = self.feature_extractor(batch["image"])
        feature_vector = torch.hstack(list(layer_outputs.values())).detach()
        batch["pred_scores"] = self.dfm_model.score(feature_vector.view(feature_vector.shape[:2]))

        return batch
