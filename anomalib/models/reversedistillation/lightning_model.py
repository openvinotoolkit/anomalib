"""Anomaly Detection via Reverse Distillation from One-Class Embedding.

https://arxiv.org/abs/2201.10703v2
"""

# Copyright (C) 2022 Intel Corporation
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

import logging
from typing import Dict, List, Tuple, Union

from omegaconf import DictConfig, ListConfig
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from torch import Tensor, optim

from anomalib.models.components import AnomalyModule

from .torch_model import ReverseDistillationModel

logger = logging.getLogger(__name__)


@MODEL_REGISTRY
class Reversedistillation(AnomalyModule):
    """PL Lightning Module for Reverse Distillation Algorithm.

    Args:
        input_size (Tuple[int, int]): Size of model input
        backbone (str): Backbone of CNN network
        layers (List[str]): Layers to extract features from the backbone CNN
        beta1 (float): Beta1 of the Adam optimizer
        beta2 (float): Beta2 of the Adam Optimizer
    """

    def __init__(self, input_size: Tuple[int, int], backbone: str, layers: List[str]):
        super().__init__()
        logger.info("Initializing Reverse Distillation Lightning model.")
        self.model = ReverseDistillationModel(backbone=backbone, layers=layers, input_size=input_size)

    def training_step(self, batch, _) -> Dict[str, Tensor]:
        """Training Step of Reverse Distillation Model.

        Features are extracted from three layers of the Encoder model. These are passed to the bottleneck layer
        that are passed to the decoder network. The loss is then calculated based on the cosine similarity between the
        encoder and decoder features.

        Args:
          batch (Tensor): Input batch
          _: Index of the batch.

        Returns:
          Feature Map
        """
        loss = self.model.get_loss(batch["image"])
        return {"loss": loss}

    def validation_step(self, batch, _):
        """Validation Step of Reverse Distillation Model.

        Similar to the training step, encoder/decoder features are extracted from the CNN for each batch, and
        anomaly map is computed.

        Args:
          batch (Tensor): Input batch
          _: Index of the batch.

        Returns:
          Dictionary containing images, anomaly maps, true labels and masks.
          These are required in `validation_epoch_end` for feature concatenation.
        """
        batch["anomaly_maps"] = self.model(batch["image"])
        return batch


class ReversedistillationLightning(Reversedistillation):
    """PL Lightning Module for Reverse Distillation Algorithm.

    Args:
        hparams(Union[DictConfig, ListConfig]): Model parameters
    """

    def __init__(self, hparams: Union[DictConfig, ListConfig]):
        super().__init__(
            input_size=hparams.model.input_size, backbone=hparams.model.backbone, layers=hparams.model.layers
        )
        self.hparams: Union[DictConfig, ListConfig]  # type: ignore
        self.save_hyperparameters(hparams)

    def configure_callbacks(self):
        """Configure model-specific callbacks.

        Note:
            This method is used for the existing CLI.
            When PL CLI is introduced, configure callback method will be
                deprecated, and callbacks will be configured from either
                config.yaml file or from CLI.
        """
        early_stopping = EarlyStopping(
            monitor=self.hparams.model.early_stopping.metric,
            patience=self.hparams.model.early_stopping.patience,
            mode=self.hparams.model.early_stopping.mode,
        )
        return [early_stopping]

    def configure_optimizers(self):
        """Configures optimizers for decoder and bottleneck.

        Note:
            This method is used for the existing CLI.
            When PL CLI is introduced, configure optimizers method will be
                deprecated, and optimizers will be configured from either
                config.yaml file or from CLI.

        Returns:
            Optimizer: Adam optimizer for each decoder
        """
        return optim.Adam(
            params=list(self.model.decoder.parameters()) + list(self.model.bottleneck.parameters()),
            lr=self.hparams.model.lr,
            betas=(self.hparams.model.beta1, self.hparams.model.beta2),
        )
