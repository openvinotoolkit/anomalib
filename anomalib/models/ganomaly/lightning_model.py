"""GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training.

https://arxiv.org/abs/1805.06725
"""

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

import logging
from typing import Dict, List, Union

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.callbacks import EarlyStopping
from torch import Tensor, optim

from anomalib.data.utils.image import pad_nextpow2
from anomalib.models.components import AnomalyModule

from .torch_model import GanomalyModel

logger = logging.getLogger(__name__)


class GanomalyLightning(AnomalyModule):
    """PL Lightning Module for the GANomaly Algorithm.

    Args:
        hparams (Union[DictConfig, ListConfig]): Model parameters
    """

    def __init__(self, hparams: Union[DictConfig, ListConfig]):
        super().__init__(hparams)
        logger.info("Initializing Ganomaly Lightning model.")

        self.model: GanomalyModel = GanomalyModel(
            input_size=hparams.model.input_size,
            num_input_channels=3,
            n_features=hparams.model.n_features,
            latent_vec_size=hparams.model.latent_vec_size,
            extra_layers=hparams.model.extra_layers,
            add_final_conv_layer=hparams.model.add_final_conv,
            wadv=self.hparams.model.wadv,
            wcon=self.hparams.model.wcon,
            wenc=self.hparams.model.wenc,
        )

        self.real_label = torch.ones(size=(self.hparams.dataset.train_batch_size,), dtype=torch.float32)
        self.fake_label = torch.zeros(size=(self.hparams.dataset.train_batch_size,), dtype=torch.float32)

        self.min_scores: Tensor = torch.tensor(float("inf"), dtype=torch.float32)  # pylint: disable=not-callable
        self.max_scores: Tensor = torch.tensor(float("-inf"), dtype=torch.float32)  # pylint: disable=not-callable

    def _reset_min_max(self):
        """Resets min_max scores."""
        self.min_scores = torch.tensor(float("inf"), dtype=torch.float32)  # pylint: disable=not-callable
        self.max_scores = torch.tensor(float("-inf"), dtype=torch.float32)  # pylint: disable=not-callable

    def configure_callbacks(self):
        """Configure model-specific callbacks."""
        early_stopping = EarlyStopping(
            monitor=self.hparams.model.early_stopping.metric,
            patience=self.hparams.model.early_stopping.patience,
            mode=self.hparams.model.early_stopping.mode,
        )
        return [early_stopping]

    def configure_optimizers(self) -> List[optim.Optimizer]:
        """Configure optimizers for generator and discriminator.

        Returns:
            List[optim.Optimizer]: Adam optimizers for discriminator and generator.
        """
        optimizer_d = optim.Adam(
            self.model.discriminator.parameters(),
            lr=self.hparams.model.lr,
            betas=(self.hparams.model.beta1, self.hparams.model.beta2),
        )
        optimizer_g = optim.Adam(
            self.model.generator.parameters(),
            lr=self.hparams.model.lr,
            betas=(self.hparams.model.beta1, self.hparams.model.beta2),
        )
        return [optimizer_d, optimizer_g]

    def training_step(self, batch, _, optimizer_idx):  # pylint: disable=arguments-differ
        """Training step.

        Args:
            batch (Dict): Input batch containing images.
            optimizer_idx (int): Optimizer which is being called for current training step.

        Returns:
            Dict[str, Tensor]: Loss
        """
        images = batch["image"]
        padded_images = pad_nextpow2(images)
        loss: Dict[str, Tensor]

        # Discriminator
        if optimizer_idx == 0:
            # forward pass
            loss_discriminator = self.model.get_discriminator_loss(padded_images)
            loss = {"loss": loss_discriminator}

        # Generator
        else:
            # forward pass
            loss_generator = self.model.get_generator_loss(padded_images)

            loss = {"loss": loss_generator}

        return loss

    def on_validation_start(self) -> None:
        """Reset min and max values for current validation epoch."""
        self._reset_min_max()
        return super().on_validation_start()

    def validation_step(self, batch, _) -> Dict[str, Tensor]:  # type: ignore # pylint: disable=arguments-differ
        """Update min and max scores from the current step.

        Args:
            batch (Dict[str, Tensor]): Predicted difference between z and z_hat.

        Returns:
            Dict[str, Tensor]: batch
        """
        batch["pred_scores"] = self.model(batch["image"])
        self.max_scores = max(self.max_scores, torch.max(batch["pred_scores"]))
        self.min_scores = min(self.min_scores, torch.min(batch["pred_scores"]))
        return batch

    def validation_epoch_end(self, outputs):
        """Normalize outputs based on min/max values."""
        logger.info("Normalizing validation outputs based on min/max values.")
        for prediction in outputs:
            prediction["pred_scores"] = self._normalize(prediction["pred_scores"])
        super().validation_epoch_end(outputs)
        return outputs

    def on_test_start(self) -> None:
        """Reset min max values before test batch starts."""
        self._reset_min_max()
        return super().on_test_start()

    def test_step(self, batch, _):
        """Update min and max scores from the current step."""
        super().test_step(batch, _)
        self.max_scores = max(self.max_scores, torch.max(batch["pred_scores"]))
        self.min_scores = min(self.min_scores, torch.min(batch["pred_scores"]))
        return batch

    def test_epoch_end(self, outputs):
        """Normalize outputs based on min/max values."""
        logger.info("Normalizing test outputs based on min/max values.")
        for prediction in outputs:
            prediction["pred_scores"] = self._normalize(prediction["pred_scores"])
        super().test_epoch_end(outputs)
        return outputs

    def _normalize(self, scores: Tensor) -> Tensor:
        """Normalize the scores based on min/max of entire dataset.

        Args:
            scores (Tensor): Un-normalized scores.

        Returns:
            Tensor: Normalized scores.
        """
        scores = (scores - self.min_scores.to(scores.device)) / (
            self.max_scores.to(scores.device) - self.min_scores.to(scores.device)
        )
        return scores
