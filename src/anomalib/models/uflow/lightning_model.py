"""U-Flow: A U-shaped Normalizing Flow for Anomaly Detection with Unsupervised Threshold.

https://arxiv.org/pdf/2211.12353.pdf
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from anomalib.models.components import AnomalyModule
from anomalib.models.uflow.loss import UFlowLoss
from anomalib.models.uflow.torch_model import UflowModel

__all__ = ["Uflow", "UflowLightning"]


class Uflow(AnomalyModule):
    """PL Lightning Module for the UFLOW algorithm."""

    def __init__(
        self,
        input_size: tuple[int, int],
        backbone: str,
        flow_steps: int = 4,
        affine_clamp: float = 2.0,
        affine_subnet_channels_ratio: float = 1.0,
        permute_soft: bool = False
    ) -> None:
        super().__init__()
        self.model: UflowModel = UflowModel(
            input_size=input_size,
            backbone=backbone,
            flow_steps=flow_steps,
            affine_clamp=affine_clamp,
            affine_subnet_channels_ratio=affine_subnet_channels_ratio,
            permute_soft=permute_soft,
        )
        self.loss = UFlowLoss()

    def step(self, batch):
        z, ljd = self.model(batch['image'])
        loss = self.loss(z, ljd)
        return {"loss": loss}

    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        losses = self.step(batch)
        self.log_dict({"loss": losses['loss']}, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return losses

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        anomaly_maps = self.model(batch["image"])
        batch["anomaly_maps"] = anomaly_maps
        return batch


class UflowLightning(Uflow):
    """PL Lightning Module for the UFLOW algorithm.

    Args:
        hparams (DictConfig | ListConfig): Model params
    """

    def __init__(self, hparams: DictConfig | ListConfig) -> None:
        super().__init__(
            input_size=hparams.model.input_size,
            backbone=hparams.model.backbone,
            flow_steps=hparams.model.flow_steps,
            affine_clamp=hparams.model.affine_clamp,
            affine_subnet_channels_ratio=hparams.model.affine_subnet_channels_ratio,
            permute_soft=hparams.model.permute_soft
        )
        self.lr = hparams.model.lr
        self.weight_decay = hparams.model.weight_decay
        self.hparams: DictConfig | ListConfig  # type: ignore
        self.save_hyperparameters(hparams)

    def configure_callbacks(self) -> list[EarlyStopping]:
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
        def get_total_number_of_iterations():
            try:
                self.trainer.reset_train_dataloader()
                number_of_training_examples = len(self.trainer.train_dataloader.dataset)
                batch_size = self.trainer.train_dataloader.loaders.batch_size
                drop_last = 1 * self.trainer.train_dataloader.loaders.drop_last
                iterations_per_epoch = number_of_training_examples // batch_size + 1 - drop_last
                total_iterations = iterations_per_epoch * (self.trainer.max_epochs - 1)
            except:
                total_iterations = 25000
            return total_iterations

        # Optimizer
        optimizer = torch.optim.Adam(
            [{"params": self.parameters(), "initial_lr": self.lr}],
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # Scheduler for slowly reducing learning rate
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1., end_factor=0.4, total_iters=get_total_number_of_iterations()
        )
        return [optimizer], [scheduler]
