"""PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization.

Paper https://arxiv.org/abs/2011.08785
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from anomalib.models.components import AnomalyModule
from anomalib.models.padim.torch_model import PadimModel

logger = logging.getLogger(__name__)

__all__ = ["Padim", "PadimLightning"]


class Padim(AnomalyModule):
    """PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization.

    Args:
        layers (list[str]): Layers to extract features from the backbone CNN
        input_size (tuple[int, int]): Size of the model input.
        backbone (str): Backbone CNN network
        pretrained_weights (str, optional): Path to pretrained weights. Defaults to None.
        tied_covariance (bool): Whether to use tied covariance matrix. Defaults to False.
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
        n_features (int, optional): Number of features to retain in the dimension reduction step.
                                Default values from the paper are available for: resnet18 (100), wide_resnet50_2 (550).
    """

    def __init__(
        self,
        layers: list[str],
        input_size: tuple[int, int],
        backbone: str,
        pretrained_weights: str | None = None,
        tied_covariance: bool = False,
        pre_trained: bool = True,
        n_features: int | None = None,
    ) -> None:
        super().__init__()

        self.layers = layers
        self.model: PadimModel = PadimModel(
            input_size=input_size,
            backbone=backbone,
            pre_trained=pre_trained,
            layers=layers,
            pretrained_weights=pretrained_weights,
            tied_covariance=tied_covariance,
            n_features=n_features,
        ).eval()

        self.stats: list[Tensor] = []
        self.embeddings: list[Tensor] = []
        self.automatic_optimization = False

    @staticmethod
    def configure_optimizers() -> None:  # pylint: disable=arguments-differ
        """PADIM doesn't require optimization, therefore returns no optimizers."""
        return None

    def on_train_epoch_start(self) -> None:
        self.embeddings = []
        self.stats = []
        return super().on_train_epoch_start()

    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> None:
        """Training Step of PADIM. For each batch, hierarchical features are extracted from the CNN.

        Args:
            batch (dict[str, str | Tensor]): Batch containing image filename, image, label and mask
            _batch_idx: Index of the batch.

        Returns:
            Hierarchical feature map
        """
        del args, kwargs  # These variables are not used.

        self.model.feature_extractor.eval()
        embedding, _ = self.model(batch["image"])

        # NOTE: `self.embedding` appends each batch embedding to
        #   store the training set embedding. We manually append these
        #   values mainly due to the new order of hooks introduced after PL v1.4.0
        #   https://github.com/PyTorchLightning/pytorch-lightning/pull/7357
        if embedding.dtype in [torch.float16, torch.bfloat16]:
            self.embeddings.append(embedding)
        else:
            self.embeddings.append(embedding.cpu())
        zero_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
        return {"loss": zero_loss}

    def on_validation_start(self) -> None:
        """Fit a Gaussian to the embedding collected from the training set."""
        # NOTE: Previous anomalib versions fit Gaussian at the end of the epoch.
        #   This is not possible anymore with PyTorch Lightning v1.4.0 since validation
        #   is run within train epoch.
        if len(self.embeddings) == 0:
            logger.warning("No embeddings were extracted from the training set. Skipping Gaussian fitting.")
            return

        logger.info("Aggregating the embedding extracted from the training set.")

        embeddings = torch.vstack(self.embeddings)

        logger.info("Fitting a Gaussian to the embedding collected from the training set.")
        self.stats = self.model.gaussian.fit(embeddings)
        self.model.gaussian.to(self.device)

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Validation Step of PADIM.

        Similar to the training step, hierarchical features are extracted from the CNN for each batch.

        Args:
            batch (dict[str, str | Tensor]): Input batch

        Returns:
            Dictionary containing images, features, true labels and masks.
            These are required in `validation_epoch_end` for feature concatenation.
        """
        del args, kwargs  # These variables are not used.

        batch["anomaly_maps"], _ = self.model(batch["image"])
        return batch


class PadimLightning(Padim):
    """PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization.

    Args:
        hparams (Union[DictConfig, ListConfig]): Model params
        backbone: optional, override hparams.model.backbone. Can be both a string or a nn.Module
    """

    def __init__(self, hparams: (DictConfig | ListConfig), backbone: str | torch.nn.Module | None = None):
        if backbone is None:
            backbone = hparams.model.backbone

        super().__init__(
            input_size=hparams.model.input_size,
            layers=hparams.model.layers,
            backbone=backbone,
            pretrained_weights=getattr(hparams.model, "pretrained_weights", None),
            tied_covariance=getattr(hparams.model, "tied_covariance", False),
            pre_trained=getattr(hparams.model, "pre_trained", True),
            n_features=getattr(hparams.model, "n_features", None),
        )
        self.hparams: DictConfig | ListConfig  # type: ignore
        self.save_hyperparameters(hparams)
