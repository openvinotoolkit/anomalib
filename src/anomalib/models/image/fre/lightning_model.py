"""FRE: Feature-Reconstruction Error.

https://papers.bmvc2023.org/0614.pdf
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim

from anomalib import LearningType
from anomalib.models.components import AnomalyModule

from .torch_model import FREModel

logger = logging.getLogger(__name__)


class Fre(AnomalyModule):
    """FRE: Feature-reconstruction error using Tied AutoEncoder.

    Args:
        backbone (str): Backbone CNN network
            Defaults to ``resnet50``.
        layer (str): Layer to extract features from the backbone CNN
            Defaults to ``layer3``.
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
            Defaults to ``True``.
        pooling_kernel_size (int, optional): Kernel size to pool features extracted from the CNN.
            Defaults to ``2``.
        input_dim (int, optional): Dimension of feature at output of layer specified in layer.
            Defaults to ``65536``.
        latent_dim (int, optional): Reduced size of feature after applying dimensionality reduction
            via shallow linear autoencoder.
            Defaults to ``220``.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        layer: str = "layer3",
        pre_trained: bool = True,
        pooling_kernel_size: int = 2,
        input_dim: int = 65536,
        latent_dim: int = 220,
    ) -> None:
        super().__init__()

        self.model: FREModel = FREModel(
            backbone=backbone,
            pre_trained=pre_trained,
            layer=layer,
            pooling_kernel_size=pooling_kernel_size,
            input_dim=input_dim,
            latent_dim=latent_dim,
        )
        self.loss_fn = torch.nn.MSELoss()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers.

        Returns:
            Optimizer: Adam optimizer
        """
        return optim.Adam(params=self.model.fre_model.parameters(), lr=1e-3)

    def training_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Perform the training step of FRE.

        For each batch, features are extracted from the CNN.

        Args:
            batch (dict[str, str | torch.Tensor]): Input batch
            args: Arguments.
            kwargs: Keyword arguments.

        Returns:
          Deep CNN features.
        """
        del args, kwargs  # These variables are not used.
        features_in, features_out, _ = self.model.get_features(batch["image"])
        loss = self.loss_fn(features_in, features_out)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Perform the validation step of FRE.

        Similar to the training step, features are extracted from the CNN for each batch.

        Args:
          batch (dict[str, str | torch.Tensor]): Input batch
          args: Arguments.
          kwargs: Keyword arguments.

        Returns:
          Dictionary containing FRE anomaly scores and anomaly maps.
        """
        del args, kwargs  # These variables are not used.

        batch["pred_scores"], batch["anomaly_maps"] = self.model(batch["image"])
        return batch

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return FRE-specific trainer arguments."""
        return {"gradient_clip_val": 0, "max_epochs": 220, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS
