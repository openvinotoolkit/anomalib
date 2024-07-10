"""GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training.

https://arxiv.org/abs/1805.06725
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim

from anomalib import LearningType
from anomalib.models.components import AnomalyModule

from .loss import DiscriminatorLoss, GeneratorLoss
from .torch_model import GanomalyModel

logger = logging.getLogger(__name__)


class Ganomaly(AnomalyModule):
    """PL Lightning Module for the GANomaly Algorithm.

    Args:
        batch_size (int): Batch size.
            Defaults to ``32``.
        n_features (int): Number of features layers in the CNNs.
            Defaults to ``64``.
        latent_vec_size (int): Size of autoencoder latent vector.
            Defaults to ``100``.
        extra_layers (int, optional): Number of extra layers for encoder/decoder.
            Defaults to ``0``.
        add_final_conv_layer (bool, optional): Add convolution layer at the end.
            Defaults to ``True``.
        wadv (int, optional): Weight for adversarial loss.
            Defaults to ``1``.
        wcon (int, optional): Image regeneration weight.
            Defaults to ``50``.
        wenc (int, optional): Latent vector encoder weight.
            Defaults to ``1``.
        lr (float, optional): Learning rate.
            Defaults to ``0.0002``.
        beta1 (float, optional): Adam beta1.
            Defaults to ``0.5``.
        beta2 (float, optional): Adam beta2.
            Defaults to ``0.999``.
    """

    def __init__(
        self,
        batch_size: int = 32,
        n_features: int = 64,
        latent_vec_size: int = 100,
        extra_layers: int = 0,
        add_final_conv_layer: bool = True,
        wadv: int = 1,
        wcon: int = 50,
        wenc: int = 1,
        lr: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.latent_vec_size = latent_vec_size
        self.extra_layers = extra_layers
        self.add_final_conv_layer = add_final_conv_layer

        self.real_label = torch.ones(size=(batch_size,), dtype=torch.float32)
        self.fake_label = torch.zeros(size=(batch_size,), dtype=torch.float32)

        self.min_scores: torch.Tensor = torch.tensor(float("inf"), dtype=torch.float32)  # pylint: disable=not-callable
        self.max_scores: torch.Tensor = torch.tensor(float("-inf"), dtype=torch.float32)  # pylint: disable=not-callable

        self.generator_loss = GeneratorLoss(wadv, wcon, wenc)
        self.discriminator_loss = DiscriminatorLoss()
        self.automatic_optimization = False

        # TODO(ashwinvaidya17): LR should be part of optimizer in config.yaml!
        # CVS-122670
        self.learning_rate = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.model: GanomalyModel

    def _setup(self) -> None:
        if self.input_size is None:
            msg = "GANomaly needs input size to build torch model."
            raise ValueError(msg)

        self.model = GanomalyModel(
            input_size=self.input_size,
            num_input_channels=3,
            n_features=self.n_features,
            latent_vec_size=self.latent_vec_size,
            extra_layers=self.extra_layers,
            add_final_conv_layer=self.add_final_conv_layer,
        )

    def _reset_min_max(self) -> None:
        """Reset min_max scores."""
        self.min_scores = torch.tensor(float("inf"), dtype=torch.float32)  # pylint: disable=not-callable
        self.max_scores = torch.tensor(float("-inf"), dtype=torch.float32)  # pylint: disable=not-callable

    def configure_optimizers(self) -> list[optim.Optimizer]:
        """Configure optimizers for each decoder.

        Returns:
            Optimizer: Adam optimizer for each decoder
        """
        optimizer_d = optim.Adam(
            self.model.discriminator.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )
        optimizer_g = optim.Adam(
            self.model.generator.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )
        return [optimizer_d, optimizer_g]

    def training_step(
        self,
        batch: dict[str, str | torch.Tensor],
        batch_idx: int,
    ) -> STEP_OUTPUT:
        """Perform the training step.

        Args:
            batch (dict[str, str | torch.Tensor]): Input batch containing images.
            batch_idx (int): Batch index.
            optimizer_idx (int): Optimizer which is being called for current training step.

        Returns:
            STEP_OUTPUT: Loss
        """
        del batch_idx  # `batch_idx` variables is not used.
        d_opt, g_opt = self.optimizers()

        # forward pass
        padded, fake, latent_i, latent_o = self.model(batch["image"])
        pred_real, _ = self.model.discriminator(padded)

        # generator update
        pred_fake, _ = self.model.discriminator(fake)
        g_loss = self.generator_loss(latent_i, latent_o, padded, fake, pred_real, pred_fake)

        g_opt.zero_grad()
        self.manual_backward(g_loss, retain_graph=True)
        g_opt.step()

        # discrimator update
        pred_fake, _ = self.model.discriminator(fake.detach())
        d_loss = self.discriminator_loss(pred_real, pred_fake)

        d_opt.zero_grad()
        self.manual_backward(d_loss)
        d_opt.step()

        self.log_dict(
            {"generator_loss": g_loss.item(), "discriminator_loss": d_loss.item()},
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"generator_loss": g_loss, "discriminator_loss": d_loss}

    def on_validation_start(self) -> None:
        """Reset min and max values for current validation epoch."""
        self._reset_min_max()
        return super().on_validation_start()

    def validation_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Update min and max scores from the current step.

        Args:
            batch (dict[str, str | torch.Tensor]): Predicted difference between z and z_hat.
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            (STEP_OUTPUT): Output predictions.
        """
        del args, kwargs  # Unused arguments.

        batch["pred_scores"] = self.model(batch["image"])
        self.max_scores = max(self.max_scores, torch.max(batch["pred_scores"]))
        self.min_scores = min(self.min_scores, torch.min(batch["pred_scores"]))
        return batch

    def on_validation_batch_end(
        self,
        outputs: STEP_OUTPUT,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Normalize outputs based on min/max values."""
        outputs["pred_scores"] = self._normalize(outputs["pred_scores"])
        super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx=dataloader_idx)

    def on_test_start(self) -> None:
        """Reset min max values before test batch starts."""
        self._reset_min_max()
        return super().on_test_start()

    def test_step(self, batch: dict[str, str | torch.Tensor], batch_idx: int, *args, **kwargs) -> STEP_OUTPUT:
        """Update min and max scores from the current step."""
        del args, kwargs  # Unused arguments.

        super().test_step(batch, batch_idx)
        self.max_scores = max(self.max_scores, torch.max(batch["pred_scores"]))
        self.min_scores = min(self.min_scores, torch.min(batch["pred_scores"]))
        return batch

    def on_test_batch_end(
        self,
        outputs: STEP_OUTPUT,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Normalize outputs based on min/max values."""
        outputs["pred_scores"] = self._normalize(outputs["pred_scores"])
        super().on_test_batch_end(outputs, batch, batch_idx, dataloader_idx=dataloader_idx)

    def _normalize(self, scores: torch.Tensor) -> torch.Tensor:
        """Normalize the scores based on min/max of entire dataset.

        Args:
            scores (torch.Tensor): Un-normalized scores.

        Returns:
            Tensor: Normalized scores.
        """
        return (scores - self.min_scores.to(scores.device)) / (
            self.max_scores.to(scores.device) - self.min_scores.to(scores.device)
        )

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return GANomaly trainer arguments."""
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS
