"""Region Based Anomaly Detection With Real-Time Training and Analysis."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Union

from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from torch import Tensor

from anomalib.models.components import AnomalyModule

from .torch_model import RKdeModel

logger = logging.getLogger(__name__)


@MODEL_REGISTRY
class RKde(AnomalyModule):
    """Region Based Anomaly Detection With Real-Time Training and Analysis.

    Args:
        backbone (str): Pre-trained model backbone.
        pre_trained (bool, optional): Boolean to check whether to use a
            pre_trained backbone.
        max_training_points (int, optional): Number of training points to fit
            the KDE model. Defaults to 40000.
        pre_processing (str, optional): Preprocess features before passing to
            KDE. Options are between `norm` and `scale`. Defaults to "scale".
        n_components (int, optional): Number of PCA components. Defaults to 16.
        threshold_steepness (float, optional): Controls how quickly the value
            saturates around zero. Defaults to 0.05.
        threshold_offset (float, optional): Offset of the density function from
            0. Defaults to 12.0.
    """

    def __init__(
        self,
        region_extractor_stage: str = "rcnn",
        min_box_size: int = 25,
        iou_threshold: float = 0.3,
        n_pca_components: int = 16,
        max_training_points: int = 40000,
        pre_processing: str = "scale",
        threshold_steepness: float = 0.05,
        threshold_offset: int = 12,
    ):
        super().__init__()

        self.model: RKdeModel = RKdeModel(
            region_extractor_stage=region_extractor_stage,
            min_box_size=min_box_size,
            iou_threshold=iou_threshold,
            n_pca_components=n_pca_components,
            pre_processing=pre_processing,
            filter_count=max_training_points,
            threshold_steepness=threshold_steepness,
            threshold_offset=threshold_offset,
        )
        self.embeddings: List[Tensor] = []

    @staticmethod
    def configure_optimizers():  # pylint: disable=arguments-differ
        """RKde doesn't require optimization, therefore returns no optimizers."""
        return None

    def training_step(self, batch, _batch_idx):  # pylint: disable=arguments-differ
        """Training Step of RKde. For each batch, features are extracted from the CNN.

        Args:
            batch (Dict[str, Any]): Batch containing image filename, image, label and mask
            _batch_idx: Index of the batch.

        Returns:
          Deep CNN features.
        """
        embedding = self.model.get_features(batch["image"]).squeeze()
        self.embeddings.append(embedding)

    def on_validation_start(self) -> None:
        """Fit a KDE Model to the embedding collected from the training set."""
        logger.info("Fitting a KDE model to the embedding collected from the training set.")
        self.model.fit(self.embeddings)

    def validation_step(self, batch, _):  # pylint: disable=arguments-differ
        """Validation Step of RKde.

        Similar to the training step, features are extracted from the CNN for each batch.

        Args:
          batch: Input batch

        Returns:
          Dictionary containing probability, prediction and ground truth values.
        """
        batch["pred_scores"] = self.model(batch["image"])

        return batch


class RKdeLightning(RKde):
    """RKde: Deep Feature Kernel Density Estimation.

    Args:
        hparams (Union[DictConfig, ListConfig]): Model params
    """

    def __init__(self, hparams: Union[DictConfig, ListConfig]) -> None:
        super().__init__(
            layers=hparams.model.layers,
            backbone=hparams.model.backbone,
            pre_trained=hparams.model.pre_trained,
            max_training_points=hparams.model.max_training_points,
            pre_processing=hparams.model.pre_processing,
            n_components=hparams.model.n_components,
            threshold_steepness=hparams.model.threshold_steepness,
            threshold_offset=hparams.model.threshold_offset,
        )
        self.hparams: Union[DictConfig, ListConfig]  # type: ignore
        self.save_hyperparameters(hparams)
