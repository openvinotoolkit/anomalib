"""Feature Reconstruction Error (FRE) Algorithm Implementation.

FRE is an anomaly detection model that uses feature reconstruction error to detect
anomalies. The model extracts features from a pre-trained CNN backbone and learns
to reconstruct them using a tied autoencoder. Anomalies are detected by measuring
the reconstruction error between the original and reconstructed features.

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models import Fre
    >>> from anomalib.engine import Engine

    >>> datamodule = MVTecAD()
    >>> model = Fre()
    >>> engine = Engine()

    >>> engine.fit(model, datamodule=datamodule)  # doctest: +SKIP
    >>> predictions = engine.predict(model, datamodule=datamodule)  # doctest: +SKIP

Paper:
    Title: FRE: Feature Reconstruction Error for Unsupervised Anomaly Detection
           and Segmentation
    URL: https://papers.bmvc2023.org/0614.pdf

See Also:
    :class:`anomalib.models.image.fre.torch_model.FREModel`:
        PyTorch implementation of the FRE model architecture.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .torch_model import FREModel

logger = logging.getLogger(__name__)


class Fre(AnomalibModule):
    """FRE: Feature-reconstruction error using Tied AutoEncoder.

    The FRE model extracts features from a pre-trained CNN backbone and learns to
    reconstruct them using a tied autoencoder. Anomalies are detected by measuring
    the reconstruction error between original and reconstructed features.

    Args:
        backbone (str): Backbone CNN network architecture.
            Defaults to ``"resnet50"``.
        layer (str): Layer name to extract features from the backbone CNN.
            Defaults to ``"layer3"``.
        pre_trained (bool, optional): Whether to use pre-trained backbone weights.
            Defaults to ``True``.
        pooling_kernel_size (int, optional): Kernel size for pooling features
            extracted from the CNN.
            Defaults to ``2``.
        input_dim (int, optional): Dimension of features at output of specified
            layer.
            Defaults to ``65536``.
        latent_dim (int, optional): Reduced feature dimension after applying
            dimensionality reduction via shallow linear autoencoder.
            Defaults to ``220``.
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
        >>> from anomalib.models import Fre
        >>> model = Fre(
        ...     backbone="resnet50",
        ...     layer="layer3",
        ...     pre_trained=True,
        ...     pooling_kernel_size=2,
        ...     input_dim=65536,
        ...     latent_dim=220,
        ... )

    See Also:
        :class:`anomalib.models.image.fre.torch_model.FREModel`:
            PyTorch implementation of the FRE model architecture.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        layer: str = "layer3",
        pre_trained: bool = True,
        pooling_kernel_size: int = 2,
        input_dim: int = 65536,
        latent_dim: int = 220,
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

        self.model: FREModel = FREModel(
            backbone=backbone,
            pre_trained=pre_trained,
            layer=layer,
            pooling_kernel_size=pooling_kernel_size,
            input_dim=input_dim,
            latent_dim=latent_dim,
        )
        self.loss_fn = torch.nn.MSELoss()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers.

        Returns:
            torch.optim.Optimizer: Adam optimizer for training the model.
        """
        return optim.Adam(params=self.model.fre_model.parameters(), lr=1e-3)

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform the training step of FRE.

        For each batch, features are extracted from the CNN backbone and
        reconstructed using the tied autoencoder. The loss is computed as the MSE
        between original and reconstructed features.

        Args:
            batch (Batch): Input batch containing images and labels.
            args: Additional arguments (unused).
            kwargs: Additional keyword arguments (unused).

        Returns:
            STEP_OUTPUT: Dictionary containing the loss value.
        """
        del args, kwargs  # These variables are not used.
        features_in, features_out, _ = self.model.get_features(batch.image)
        loss = self.loss_fn(features_in, features_out)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform the validation step of FRE.

        Similar to training, features are extracted and reconstructed. The
        reconstruction error is used to compute anomaly scores and maps.

        Args:
            batch (Batch): Input batch containing images and labels.
            args: Additional arguments (unused).
            kwargs: Additional keyword arguments (unused).

        Returns:
            STEP_OUTPUT: Dictionary containing anomaly scores and maps.
        """
        del args, kwargs  # These variables are not used.

        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return FRE-specific trainer arguments.

        Returns:
            dict[str, Any]: Dictionary of trainer arguments:
                - ``gradient_clip_val``: ``0``
                - ``max_epochs``: ``220``
                - ``num_sanity_val_steps``: ``0``
        """
        return {"gradient_clip_val": 0, "max_epochs": 220, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model (``ONE_CLASS``).
        """
        return LearningType.ONE_CLASS
