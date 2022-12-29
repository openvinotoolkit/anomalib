"""DFKDE: Deep Feature Kernel Density Estimation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Union

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from torch import Tensor

from anomalib.models.components import AnomalyModule
from anomalib.models.components.classification import FeatureScalingMethod

from .torch_model import DfkdeModel

logger = logging.getLogger(__name__)


@MODEL_REGISTRY
class Dfkde(AnomalyModule):
    """DFKDE: Deep Feature Kernel Density Estimation.

    Args:
        backbone (str): Pre-trained model backbone.
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
        max_training_points (int, optional): Number of training points to fit the KDE model.
            Defaults to 40000.
        pre_processing (str, optional): Preprocess features before passing to KDE.
            Options are between `norm` and `scale`. Defaults to "scale".
        n_components (int, optional): Number of PCA components. Defaults to 16.
        threshold_steepness (float, optional): Controls how quickly the value saturates around zero.
            Defaults to 0.05.
        threshold_offset (float, optional): Offset of the density function from 0. Defaults to 12.0.
    """

    def __init__(
        self,
        layers: List[str],
        backbone: str,
        pre_trained: bool = True,
        n_pca_components: int = 16,
        feature_scaling_method: FeatureScalingMethod = FeatureScalingMethod.SCALE,
        max_training_points: int = 40000,
    ):
        super().__init__()

        self.model = DfkdeModel(
            layers=layers,
            backbone=backbone,
            pre_trained=pre_trained,
            n_pca_components=n_pca_components,
            feature_scaling_method=feature_scaling_method,
            max_training_points=max_training_points,
        )

        self.embeddings: List[Tensor] = []

    @staticmethod
    def configure_optimizers():  # pylint: disable=arguments-differ
        """DFKDE doesn't require optimization, therefore returns no optimizers."""
        return None

    def training_step(self, batch, _batch_idx):  # pylint: disable=arguments-differ
        """Training Step of DFKDE. For each batch, features are extracted from the CNN.

        Args:
            batch (Dict[str, Any]): Batch containing image filename, image, label and mask
            _batch_idx: Index of the batch.

        Returns:
          Deep CNN features.
        """

        embedding = self.model(batch["image"])

        # NOTE: `self.embedding` appends each batch embedding to
        #   store the training set embedding. We manually append these
        #   values mainly due to the new order of hooks introduced after PL v1.4.0
        #   https://github.com/PyTorchLightning/pytorch-lightning/pull/7357
        self.embeddings.append(embedding)

    def on_validation_start(self) -> None:
        """Fit a KDE Model to the embedding collected from the training set."""
        # NOTE: Previous anomalib versions fit Gaussian at the end of the epoch.
        #   This is not possible anymore with PyTorch Lightning v1.4.0 since validation
        #   is run within train epoch.
        embeddings = torch.vstack(self.embeddings)

        logger.info("Fitting a KDE model to the embedding collected from the training set.")
        self.model.classifier.fit(embeddings)

    def validation_step(self, batch, _):  # pylint: disable=arguments-differ
        """Validation Step of DFKDE.

        Similar to the training step, features are extracted from the CNN for each batch.

        Args:
          batch: Input batch

        Returns:
          Dictionary containing probability, prediction and ground truth values.
        """
        batch["pred_scores"] = self.model(batch["image"])

        return batch


class DfkdeLightning(Dfkde):
    """DFKDE: Deep Feature Kernel Density Estimation.

    Args:
        hparams (Union[DictConfig, ListConfig]): Model params
    """

    def __init__(self, hparams: Union[DictConfig, ListConfig]) -> None:
        super().__init__(
            layers=hparams.model.layers,
            backbone=hparams.model.backbone,
            pre_trained=hparams.model.pre_trained,
            n_pca_components=hparams.model.n_pca_components,
            feature_scaling_method=FeatureScalingMethod(hparams.model.feature_scaling_method),
            max_training_points=hparams.model.max_training_points,
        )
        self.hparams: Union[DictConfig, ListConfig]  # type: ignore
        self.save_hyperparameters(hparams)
