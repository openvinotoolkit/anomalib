"""DFKDE: Deep Feature Kernel Density Estimation."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import AUROC, Evaluator, F1Score
from anomalib.models.components import AnomalibModule, MemoryBankMixin
from anomalib.models.components.classification import FeatureScalingMethod
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor

from .torch_model import DfkdeModel

logger = logging.getLogger(__name__)


class Dfkde(MemoryBankMixin, AnomalibModule):
    """DFKDE: Deep Feature Kernel Density Estimation.

    Args:
        backbone (str): Pre-trained model backbone.
            Defaults to ``"resnet18"``.
        layers (Sequence[str], optional): Layers to extract features from.
            Defaults to ``("layer4",)``.
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
            Defaults to ``True``.
        n_pca_components (int, optional): Number of PCA components.
            Defaults to ``16``.
        feature_scaling_method (FeatureScalingMethod, optional): Feature scaling method.
            Defaults to ``FeatureScalingMethod.SCALE``.
        max_training_points (int, optional): Number of training points to fit the KDE model.
            Defaults to ``40000``.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: Sequence[str] = ("layer4",),
        pre_trained: bool = True,
        n_pca_components: int = 16,
        feature_scaling_method: FeatureScalingMethod = FeatureScalingMethod.SCALE,
        max_training_points: int = 40000,
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
    ) -> None:
        super().__init__(pre_processor=pre_processor, post_processor=post_processor, evaluator=evaluator)

        self.model = DfkdeModel(
            layers=layers,
            backbone=backbone,
            pre_trained=pre_trained,
            n_pca_components=n_pca_components,
            feature_scaling_method=feature_scaling_method,
            max_training_points=max_training_points,
        )

        self.embeddings: list[torch.Tensor] = []

    @staticmethod
    def configure_optimizers() -> None:  # pylint: disable=arguments-differ
        """DFKDE doesn't require optimization, therefore returns no optimizers."""
        return

    def training_step(self, batch: Batch, *args, **kwargs) -> None:
        """Perform the training step of DFKDE. For each batch, features are extracted from the CNN.

        Args:
            batch (batch: Batch): Batch containing image filename, image, label and mask
            args: Arguments.
            kwargs: Keyword arguments.

        Returns:
          Deep CNN features.
        """
        del args, kwargs  # These variables are not used.

        embedding = self.model(batch.image)
        self.embeddings.append(embedding)

    def fit(self) -> None:
        """Fit a KDE Model to the embedding collected from the training set."""
        embeddings = torch.vstack(self.embeddings)

        logger.info("Fitting a KDE model to the embedding collected from the training set.")
        self.model.classifier.fit(embeddings)

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform the validation step of DFKDE.

        Similar to the training step, features are extracted from the CNN for each batch.

        Args:
            batch (Batch): Input batch
            args: Arguments.
            kwargs: Keyword arguments.

        Returns:
            Dictionary containing probability, prediction and ground truth values.
        """
        del args, kwargs  # These variables are not used.

        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return DFKDE-specific trainer arguments."""
        return {"gradient_clip_val": 0, "max_epochs": 1, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS

    @staticmethod
    def configure_evaluator() -> Evaluator:
        """Default evaluator for DFKE."""
        image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
        image_f1score = F1Score(fields=["pred_label", "gt_label"], prefix="image_")
        test_metrics = [image_auroc, image_f1score]
        return Evaluator(test_metrics=test_metrics)
