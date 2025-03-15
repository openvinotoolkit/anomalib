"""PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization.

This model implements the PaDiM algorithm for anomaly detection and localization.
PaDiM models the distribution of patch embeddings at each spatial location using
multivariate Gaussian distributions.

The model extracts features from multiple layers of pretrained CNN backbones to
capture both semantic and low-level visual information. During inference, it
computes Mahalanobis distances between test patch embeddings and their
corresponding reference distributions.

Paper: https://arxiv.org/abs/2011.08785

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models.image.padim import Padim
    >>> from anomalib.engine import Engine

    >>> # Initialize model and data
    >>> datamodule = MVTecAD()
    >>> model = Padim(
    ...     backbone="resnet18",
    ...     layers=["layer1", "layer2", "layer3"],
    ...     pre_trained=True
    ... )

    >>> # Train using the Engine
    >>> engine = Engine()
    >>> engine.fit(model=model, datamodule=datamodule)

    >>> # Get predictions
    >>> predictions = engine.predict(model=model, datamodule=datamodule)

See Also:
    - :class:`anomalib.models.image.padim.torch_model.PadimModel`:
        PyTorch implementation of the PaDiM model architecture
    - :class:`anomalib.models.image.padim.anomaly_map.AnomalyMapGenerator`:
        Anomaly map generation for PaDiM using Mahalanobis distance
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule, MemoryBankMixin
from anomalib.post_processing import PostProcessor
from anomalib.visualization import Visualizer

from .torch_model import PadimModel

logger = logging.getLogger(__name__)

__all__ = ["Padim"]


class Padim(MemoryBankMixin, AnomalibModule):
    """PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection.

    Args:
        backbone (str): Name of the backbone CNN network. Available options are
            ``resnet18``, ``wide_resnet50_2`` etc. Defaults to ``resnet18``.
        layers (list[str]): List of layer names to extract features from the
            backbone CNN. Defaults to ``["layer1", "layer2", "layer3"]``.
        pre_trained (bool, optional): Use pre-trained backbone weights.
            Defaults to ``True``.
        n_features (int | None, optional): Number of features to retain after
            dimension reduction. Default values from paper: ``resnet18=100``,
            ``wide_resnet50_2=550``. Defaults to ``None``.
        pre_processor (PreProcessor | bool, optional): Preprocessor to apply on
            input data. Defaults to ``True``.
        post_processor (PostProcessor | bool, optional): Post processor to apply
            on model outputs. Defaults to ``True``.
        evaluator (Evaluator | bool, optional): Evaluator for computing metrics.
            Defaults to ``True``.
        visualizer (Visualizer | bool, optional): Visualizer for generating
            result images. Defaults to ``True``.

    Example:
        >>> from anomalib.models import Padim
        >>> from anomalib.data import MVTecAD
        >>> from anomalib.engine import Engine

        >>> # Initialize model and data
        >>> datamodule = MVTecAD()
        >>> model = Padim(
        ...     backbone="resnet18",
        ...     layers=["layer1", "layer2", "layer3"],
        ...     pre_trained=True
        ... )

        >>> engine = Engine()
        >>> engine.train(model=model, datamodule=datamodule)
        >>> predictions = engine.predict(model=model, datamodule=datamodule)

    Note:
        The model does not require training in the traditional sense. It fits
        Gaussian distributions to the extracted features during the training
        phase.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: list[str] = ["layer1", "layer2", "layer3"],  # noqa: B006
        pre_trained: bool = True,
        n_features: int | None = None,
        pre_processor: nn.Module | bool = True,
        post_processor: nn.Module | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        self.model: PadimModel = PadimModel(
            backbone=backbone,
            pre_trained=pre_trained,
            layers=layers,
            n_features=n_features,
        )

        self.stats: list[torch.Tensor] = []
        self.embeddings: list[torch.Tensor] = []

    @staticmethod
    def configure_optimizers() -> None:
        """PADIM doesn't require optimization, therefore returns no optimizers."""
        return

    def training_step(self, batch: Batch, *args, **kwargs) -> None:
        """Perform the training step of PADIM.

        For each batch, hierarchical features are extracted from the CNN.

        Args:
            batch (Batch): Input batch containing image and metadata
            args: Additional arguments (unused)
            kwargs: Additional keyword arguments (unused)

        Returns:
            torch.Tensor: Dummy loss tensor for Lightning compatibility
        """
        del args, kwargs  # These variables are not used.

        embedding = self.model(batch.image)
        self.embeddings.append(embedding)

        # Return a dummy loss tensor
        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def fit(self) -> None:
        """Fit a Gaussian to the embedding collected from the training set."""
        logger.info("Aggregating the embedding extracted from the training set.")
        embeddings = torch.vstack(self.embeddings)

        logger.info("Fitting a Gaussian to the embedding collected from the training set.")
        self.stats = self.model.gaussian.fit(embeddings)

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform a validation step of PADIM.

        Similar to the training step, hierarchical features are extracted from
        the CNN for each batch.

        Args:
            batch (Batch): Input batch containing image and metadata
            args: Additional arguments (unused)
            kwargs: Additional keyword arguments (unused)

        Returns:
            STEP_OUTPUT: Dictionary containing images, features, true labels
            and masks required for validation
        """
        del args, kwargs  # These variables are not used.

        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, int | float]:
        """Return PADIM trainer arguments.

        Since the model does not require training, we limit the max_epochs to 1.
        Since we need to run training epoch before validation, we also set the
        sanity steps to 0.

        Returns:
            dict[str, int | float]: Dictionary of trainer arguments
        """
        return {"max_epochs": 1, "val_check_interval": 1.0, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type (ONE_CLASS for PaDiM)
        """
        return LearningType.ONE_CLASS

    @staticmethod
    def configure_post_processor() -> PostProcessor:
        """Return the default post-processor for PADIM.

        Returns:
            PostProcessor: Default post-processor
        """
        return PostProcessor()
