"""PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization.

Paper https://arxiv.org/abs/2011.08785
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Tuple, Union

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from torch import Tensor

from anomalib.models.components import AnomalyModule
from anomalib.models.padim.torch_model import PadimModel

logger = logging.getLogger(__name__)

__all__ = ["Padim", "PadimLightning"]


@MODEL_REGISTRY
class Padim(AnomalyModule):
    """PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization.

    Args:
        layers (List[str]): Layers to extract features from the backbone CNN
        input_size (Tuple[int, int]): Size of the model input.
        backbone (str): Backbone CNN network
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
    """

    def __init__(
        self,
        layers: List[str],
        input_size: Tuple[int, int],
        backbone: str,
        pre_trained: bool = True,
    ):
        super().__init__()

        self.layers = layers
        self.model: PadimModel = PadimModel(
            input_size=input_size,
            backbone=backbone,
            pre_trained=pre_trained,
            layers=layers,
        ).eval()

        self.stats: List[Tensor] = []
        self.embeddings: List[Tensor] = []

    @staticmethod
    def configure_optimizers():  # pylint: disable=arguments-differ
        """PADIM doesn't require optimization, therefore returns no optimizers."""
        return None

    def training_step(self, batch, _batch_idx):  # pylint: disable=arguments-differ
        """Training Step of PADIM. For each batch, hierarchical features are extracted from the CNN.

        Args:
            batch (Dict[str, Any]): Batch containing image filename, image, label and mask
            _batch_idx: Index of the batch.

        Returns:
            Hierarchical feature map
        """
        self.model.feature_extractor.eval()
        embedding = self.model(batch["image"])

        # NOTE: `self.embedding` appends each batch embedding to
        #   store the training set embedding. We manually append these
        #   values mainly due to the new order of hooks introduced after PL v1.4.0
        #   https://github.com/PyTorchLightning/pytorch-lightning/pull/7357
        self.embeddings.append(embedding.cpu())

    def on_validation_start(self) -> None:
        """Fit a Gaussian to the embedding collected from the training set."""
        # NOTE: Previous anomalib versions fit Gaussian at the end of the epoch.
        #   This is not possible anymore with PyTorch Lightning v1.4.0 since validation
        #   is run within train epoch.
        logger.info("Aggregating the embedding extracted from the training set.")
        embeddings = torch.vstack(self.embeddings)

        logger.info("Fitting a Gaussian to the embedding collected from the training set.")
        self.stats = self.model.gaussian.fit(embeddings)

    def validation_step(self, batch, _):  # pylint: disable=arguments-differ
        """Validation Step of PADIM.

        Similar to the training step, hierarchical features are extracted from the CNN for each batch.

        Args:
            batch: Input batch
            _: Index of the batch.

        Returns:
            Dictionary containing images, features, true labels and masks.
            These are required in `validation_epoch_end` for feature concatenation.
        """

        batch["anomaly_maps"] = self.model(batch["image"])
        return batch


class PadimLightning(Padim):
    """PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization.

    Args:
        hparams (Union[DictConfig, ListConfig]): Model params
    """

    def __init__(self, hparams: Union[DictConfig, ListConfig]):
        super().__init__(
            input_size=hparams.model.input_size,
            layers=hparams.model.layers,
            backbone=hparams.model.backbone,
        )
        self.hparams: Union[DictConfig, ListConfig]  # type: ignore
        self.save_hyperparameters(hparams)
