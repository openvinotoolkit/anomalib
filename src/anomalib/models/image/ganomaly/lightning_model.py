"""GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training.

GANomaly is an anomaly detection model that uses a conditional GAN architecture to
learn the normal data distribution. The model consists of a generator network that
learns to reconstruct normal images, and a discriminator that helps ensure the
reconstructions are realistic.

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models import Ganomaly
    >>> from anomalib.engine import Engine

    >>> datamodule = MVTecAD()
    >>> model = Ganomaly()
    >>> engine = Engine()

    >>> engine.fit(model, datamodule=datamodule)  # doctest: +SKIP
    >>> predictions = engine.predict(model, datamodule=datamodule)  # doctest: +SKIP

Paper:
    Title: GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training
    URL: https://arxiv.org/abs/1805.06725

See Also:
    :class:`anomalib.models.image.ganomaly.torch_model.GanomalyModel`:
        PyTorch implementation of the GANomaly model architecture.
    :class:`anomalib.models.image.ganomaly.loss.GeneratorLoss`:
        Loss function for the generator network.
    :class:`anomalib.models.image.ganomaly.loss.DiscriminatorLoss`:
        Loss function for the discriminator network.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import AUROC, Evaluator, F1Score
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .loss import DiscriminatorLoss, GeneratorLoss
from .torch_model import GanomalyModel

logger = logging.getLogger(__name__)


class Ganomaly(AnomalibModule):
    """PL Lightning Module for the GANomaly Algorithm.

    The GANomaly model consists of a generator and discriminator network. The
    generator learns to reconstruct normal images while the discriminator helps
    ensure the reconstructions are realistic. Anomalies are detected by measuring
    the reconstruction error and latent space differences.

    Args:
        batch_size (int): Number of samples in each batch.
            Defaults to ``32``.
        n_features (int): Number of feature channels in CNN layers.
            Defaults to ``64``.
        latent_vec_size (int): Dimension of the latent space vectors.
            Defaults to ``100``.
        extra_layers (int, optional): Number of extra layers in encoder/decoder.
            Defaults to ``0``.
        add_final_conv_layer (bool, optional): Add convolution layer at the end.
            Defaults to ``True``.
        wadv (int, optional): Weight for adversarial loss component.
            Defaults to ``1``.
        wcon (int, optional): Weight for image reconstruction loss component.
            Defaults to ``50``.
        wenc (int, optional): Weight for latent vector encoding loss component.
            Defaults to ``1``.
        lr (float, optional): Learning rate for optimizers.
            Defaults to ``0.0002``.
        beta1 (float, optional): Beta1 parameter for Adam optimizers.
            Defaults to ``0.5``.
        beta2 (float, optional): Beta2 parameter for Adam optimizers.
            Defaults to ``0.999``.
        pre_processor (PreProcessor | bool, optional): Pre-processor to transform
            inputs before passing to model.
            Defaults to ``True``.
        post_processor (PostProcessor | bool, optional): Post-processor to generate
            predictions from model outputs.
            Defaults to ``True``.
        evaluator (Evaluator | bool, optional): Evaluator to compute metrics.
            Defaults to ``True``.
        visualizer (Visualizer | bool, optional): Visualizer to display results.
            Defaults to ``True``.

    Example:
        >>> from anomalib.models import Ganomaly
        >>> model = Ganomaly(
        ...     batch_size=32,
        ...     n_features=64,
        ...     latent_vec_size=100,
        ...     wadv=1,
        ...     wcon=50,
        ...     wenc=1,
        ... )

    See Also:
        :class:`anomalib.models.image.ganomaly.torch_model.GanomalyModel`:
            PyTorch implementation of the GANomaly model architecture.
        :class:`anomalib.models.image.ganomaly.loss.GeneratorLoss`:
            Loss function for the generator network.
        :class:`anomalib.models.image.ganomaly.loss.DiscriminatorLoss`:
            Loss function for the discriminator network.
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
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
        if self.input_size is None:
            msg = "GANomaly needs input size to build torch model."
            raise ValueError(msg)

        self.n_features = n_features
        self.latent_vec_size = latent_vec_size
        self.extra_layers = extra_layers
        self.add_final_conv_layer = add_final_conv_layer

        self.real_label = torch.ones(size=(batch_size,), dtype=torch.float32)
        self.fake_label = torch.zeros(size=(batch_size,), dtype=torch.float32)

        self.min_scores: torch.Tensor = torch.tensor(float("inf"), dtype=torch.float32)  # pylint: disable=not-callable
        self.max_scores: torch.Tensor = torch.tensor(float("-inf"), dtype=torch.float32)  # pylint: disable=not-callable

        self.model = GanomalyModel(
            input_size=self.input_size,
            num_input_channels=3,
            n_features=self.n_features,
            latent_vec_size=self.latent_vec_size,
            extra_layers=self.extra_layers,
            add_final_conv_layer=self.add_final_conv_layer,
        )

        self.generator_loss = GeneratorLoss(wadv, wcon, wenc)
        self.discriminator_loss = DiscriminatorLoss()
        self.automatic_optimization = False

        # TODO(ashwinvaidya17): LR should be part of optimizer in config.yaml!
        # CVS-122670
        self.learning_rate = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.model: GanomalyModel

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
        batch: Batch,
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
        padded, fake, latent_i, latent_o = self.model(batch.image)
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

    def validation_step(self, batch: Batch, *args, **kwargs) -> Batch:
        """Update min and max scores from the current step.

        Args:
            batch (Batch): Predicted difference between z and z_hat.
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            (STEP_OUTPUT): Output predictions.
        """
        del args, kwargs  # Unused arguments.

        predictions = self.model(batch.image)
        self.max_scores = max(self.max_scores, torch.max(predictions.pred_score))
        self.min_scores = min(self.min_scores, torch.min(predictions.pred_score))
        return batch.update(**predictions._asdict())

    def on_validation_batch_end(
        self,
        outputs: Batch,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Normalize outputs based on min/max values."""
        outputs.pred_score = self._normalize(outputs.pred_score)
        super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx=dataloader_idx)

    def on_test_start(self) -> None:
        """Reset min max values before test batch starts."""
        self._reset_min_max()
        return super().on_test_start()

    def test_step(self, batch: Batch, batch_idx: int, *args, **kwargs) -> Batch:
        """Update min and max scores from the current step."""
        del args, kwargs  # Unused arguments.

        super().test_step(batch, batch_idx)
        self.max_scores = max(self.max_scores, torch.max(batch.pred_score))
        self.min_scores = min(self.min_scores, torch.min(batch.pred_score))
        return batch

    def on_test_batch_end(
        self,
        outputs: Batch,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Normalize outputs based on min/max values."""
        outputs.pred_score = self._normalize(outputs.pred_score)
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

    @staticmethod
    def configure_evaluator() -> Evaluator:
        """Default evaluator for GANomaly."""
        image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
        image_f1score = F1Score(fields=["pred_label", "gt_label"], prefix="image_")
        test_metrics = [image_auroc, image_f1score]
        return Evaluator(test_metrics=test_metrics)
