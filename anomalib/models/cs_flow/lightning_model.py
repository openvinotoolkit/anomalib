"""Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection.

https://arxiv.org/pdf/2110.02855.pdf
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Tuple, Union

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from torch import Tensor

from anomalib.models.components import AnomalyModule

from .torch_model import CsFlowModel

logger = logging.getLogger(__name__)

__all__ = ["CsFlow", "CsFlowLightning"]


@MODEL_REGISTRY
class CsFlow(AnomalyModule):
    """Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection.

    Args:
        input_size (Tuple[int, int]): Size of the model input.
        n_coupling_blocks (int): Number of coupling blocks in the model.
        clamp (int): Clamp value for glow layer.
        num_channels (int): Number of channels in the model.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        cross_conv_hidden_channels: int,
        n_coupling_blocks: int,
        clamp: int,
        num_channels: int,
    ):
        super().__init__()
        self.model: CsFlowModel = CsFlowModel(
            input_size=input_size,
            cross_conv_hidden_channels=cross_conv_hidden_channels,
            n_coupling_blocks=n_coupling_blocks,
            clamp=clamp,
            num_channels=num_channels,
        )

    def _get_loss(self, z_dist: Tensor, jacobians: Tensor):
        """Loss function of CsFlow.

        Args:
            z_distribution (Tensor): Latent space image mappings from NF.
            jacobians (Tensor): Jacobians of the distribution

        Returns:
            Loss value
        """
        z_dist = torch.cat([z_dist[i].reshape(z_dist[i].shape[0], -1) for i in range(len(z_dist))], dim=1)
        return torch.mean(0.5 * torch.sum(z_dist**2, dim=(1,)) - jacobians) / z_dist.shape[1]

    def training_step(self, batch, _):
        """Training Step of CsFlow.

        Args:
            batch (Tensor): Input batch
            _: Index of the batch.

        Returns:
            Loss value
        """
        self.model.feature_extractor.eval()
        z_dist, jacobians = self.model(batch["image"])
        loss = self._get_loss(z_dist, jacobians)
        return {"loss": loss}

    def validation_step(self, batch, _):
        """Validation step for CS Flow."""
        anomaly_maps, anomaly_scores = self.model(batch["image"])
        batch["anomaly_maps"] = anomaly_maps
        batch["anomaly_scores"] = anomaly_scores
        return batch


class CsFlowLightning(CsFlow):
    """Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection.

    Args:
        hprams (Union[DictConfig, ListConfig]): Model params
    """

    def __init__(self, hparams: Union[DictConfig, ListConfig]):
        super().__init__(
            input_size=hparams.model.input_size,
            n_coupling_blocks=hparams.model.n_coupling_blocks,
            cross_conv_hidden_channels=hparams.model.cross_conv_hidden_channels,
            clamp=hparams.model.clamp,
            num_channels=3,
        )
        self.hparams: Union[DictConfig, ListConfig]  # type: ignore
        self.save_hyperparameters(hparams)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures optimizers.

        Note:
            This method is used for the existing CLI.
            When PL CLI is introduced, configure optimizers method will be
                deprecated, and optimizers will be configured from either
                config.yaml file or from CLI.

        Returns:
            Optimizer: Adam optimizer
        """
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.model.lr,
            eps=self.hparams.model.eps,
            weight_decay=self.hparams.model.weight_decay,
        )
