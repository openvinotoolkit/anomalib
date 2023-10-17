"""DFM: Deep Feature Modeling."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig, ListConfig
from torch import Tensor

from anomalib.models.components import AnomalyModule

from .torch_model import DFMModel

logger = logging.getLogger(__name__)


class Dfm(AnomalyModule):
    """DFM: Deep Featured Kernel Density Estimation.

    Args:
        backbone (str): Backbone CNN network
        layer (str): Layer to extract features from the backbone CNN
        input_size (tuple[int, int]): Input size for the model.
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
        pooling_kernel_size (int, optional): Kernel size to pool features extracted from the CNN.
            Defaults to 4.
        pca_level (float, optional): Ratio from which number of components for PCA are calculated.
            Defaults to 0.97.
        score_type (str, optional): Scoring type. Options are `fre` and `nll`. Defaults to "fre".
        nll: for Gaussian modeling, fre: pca feature-reconstruction error. Anomaly segmentation is
        supported with `fre` only. If using `nll`, set `task` in config.yaml to classification
    """

    def __init__(
        self,
        backbone: str,
        layer: str,
        input_size: tuple[int, int],
        pre_trained: bool = True,
        pooling_kernel_size: int = 4,
        pca_level: float = 0.97,
        score_type: str = "fre",
    ) -> None:
        super().__init__()

        self.model: DFMModel = DFMModel(
            backbone=backbone,
            pre_trained=pre_trained,
            layer=layer,
            input_size=input_size,
            pooling_kernel_size=pooling_kernel_size,
            n_comps=pca_level,
            score_type=score_type,
        )
        self.embeddings: list[Tensor] = []
        self.score_type = score_type

    @staticmethod
    def configure_optimizers() -> None:  # pylint: disable=arguments-differ
        """DFM doesn't require optimization, therefore returns no optimizers."""
        return

    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> None:
        """Perform the training step of DFM.

        For each batch, features are extracted from the CNN.

        Args:
            batch (dict[str, str | Tensor]): Input batch
            args: Arguments.
            kwargs: Keyword arguments.

        Returns:
          Deep CNN features.
        """
        del args, kwargs  # These variables are not used.

        embedding = self.model.get_features(batch["image"]).squeeze()

        # NOTE: `self.embedding` appends each batch embedding to
        #   store the training set embedding. We manually append these
        #   values mainly due to the new order of hooks introduced after PL v1.4.0
        #   https://github.com/PyTorchLightning/pytorch-lightning/pull/7357
        self.embeddings.append(embedding)

    def on_validation_start(self) -> None:
        """Fit a PCA transformation and a Gaussian model to dataset."""
        # NOTE: Previous anomalib versions fit Gaussian at the end of the epoch.
        #   This is not possible anymore with PyTorch Lightning v1.4.0 since validation
        #   is run within train epoch.
        logger.info("Aggregating the embedding extracted from the training set.")
        embeddings = torch.vstack(self.embeddings)

        logger.info("Fitting a PCA and a Gaussian model to dataset.")
        self.model.fit(embeddings)

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Perform the validation step of DFM.

        Similar to the training step, features are extracted from the CNN for each batch.

        Args:
          batch (dict[str, str | Tensor]): Input batch
          args: Arguments.
          kwargs: Keyword arguments.

        Returns:
          Dictionary containing FRE anomaly scores and anomaly maps.
        """
        del args, kwargs  # These variables are not used.

        if self.score_type == "fre":
            batch["anomaly_maps"], batch["pred_scores"] = self.model(batch["image"])
        elif self.score_type == "nll":
            batch["pred_scores"] = self.model(batch["image"])

        return batch

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return DFM-specific trainer arguments."""
        return {"gradient_clip_val": 0, "max_epochs": 1, "num_sanity_val_steps": 0}


class DfmLightning(Dfm):
    """DFM: Deep Featured Kernel Density Estimation.

    Args:
        hparams (DictConfig | ListConfig): Model params
    """

    def __init__(self, hparams: DictConfig | ListConfig) -> None:
        super().__init__(
            input_size=hparams.model.input_size,
            backbone=hparams.model.backbone,
            layer=hparams.model.layer,
            pre_trained=hparams.model.pre_trained,
            pooling_kernel_size=hparams.model.pooling_kernel_size,
            pca_level=hparams.model.pca_level,
            score_type=hparams.model.score_type,
        )
        self.hparams: DictConfig | ListConfig
        self.save_hyperparameters(hparams)
