"""Anomaly Detection via Reverse Distillation from One-Class Embedding.

This module implements the Reverse Distillation model for anomaly detection as described in
`Deng et al. (2022) <https://arxiv.org/abs/2201.10703v2>`_.

The model consists of:
- A pre-trained encoder (e.g. ResNet) that extracts multi-scale features
- A bottleneck layer that compresses features into a compact representation
- A decoder that reconstructs features back to the original feature space
- A scoring mechanism based on reconstruction error

Example:
    >>> from anomalib.models import ReverseDistillation
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.engine import Engine

    >>> # Initialize model and data
    >>> datamodule = MVTecAD()
    >>> model = ReverseDistillation(
    ...     backbone="wide_resnet50_2",
    ...     layers=["layer1", "layer2", "layer3"]
    ... )

    >>> # Train using the Engine
    >>> engine = Engine()
    >>> engine.fit(model=model, datamodule=datamodule)

    >>> # Get predictions
    >>> predictions = engine.predict(model=model, datamodule=datamodule)

See Also:
    - :class:`ReverseDistillation`: Lightning implementation of the model
    - :class:`ReverseDistillationModel`: PyTorch implementation of the model
    - :class:`ReverseDistillationLoss`: Loss function for training
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any

from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .anomaly_map import AnomalyMapGenerationMode
from .loss import ReverseDistillationLoss
from .torch_model import ReverseDistillationModel


class ReverseDistillation(AnomalibModule):
    """PL Lightning Module for Reverse Distillation Algorithm.

    Args:
        backbone (str): Backbone of CNN network
            Defaults to ``wide_resnet50_2``.
        layers (list[str]): Layers to extract features from the backbone CNN
            Defaults to ``["layer1", "layer2", "layer3"]``.
        anomaly_map_mode (AnomalyMapGenerationMode, optional): Mode to generate anomaly map.
            Defaults to ``AnomalyMapGenerationMode.ADD``.
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
            Defaults to ``True``.
        pre_processor (PreProcessor, optional): Pre-processor for the model.
            This is used to pre-process the input data before it is passed to the model.
            Defaults to ``None``.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: Sequence[str] = ("layer1", "layer2", "layer3"),
        anomaly_map_mode: AnomalyMapGenerationMode = AnomalyMapGenerationMode.ADD,
        pre_trained: bool = True,
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
        if self.input_size is None:
            msg = "Input size is required for Reverse Distillation model."
            raise ValueError(msg)

        self.backbone = backbone
        self.pre_trained = pre_trained
        self.layers = layers
        self.anomaly_map_mode = anomaly_map_mode

        self.model = ReverseDistillationModel(
            backbone=self.backbone,
            pre_trained=self.pre_trained,
            layers=self.layers,
            input_size=self.input_size,
            anomaly_map_mode=self.anomaly_map_mode,
        )
        self.loss = ReverseDistillationLoss()

    def configure_optimizers(self) -> optim.Adam:
        """Configure optimizers for decoder and bottleneck.

        Returns:
            Optimizer: Adam optimizer for each decoder
        """
        return optim.Adam(
            params=list(self.model.decoder.parameters()) + list(self.model.bottleneck.parameters()),
            lr=0.005,
            betas=(0.5, 0.99),
        )

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform a training step of Reverse Distillation Model.

        Features are extracted from three layers of the Encoder model. These are passed to the bottleneck layer
        that are passed to the decoder network. The loss is then calculated based on the cosine similarity between the
        encoder and decoder features.

        Args:
          batch (batch: Batch): Input batch
          args: Additional arguments.
          kwargs: Additional keyword arguments.

        Returns:
          Feature Map
        """
        del args, kwargs  # These variables are not used.

        loss = self.loss(*self.model(batch.image))
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform a validation step of Reverse Distillation Model.

        Similar to the training step, encoder/decoder features are extracted from the CNN for each batch, and
        anomaly map is computed.

        Args:
          batch (Batch): Input batch
          args: Additional arguments.
          kwargs: Additional keyword arguments.

        Returns:
          Dictionary containing images, anomaly maps, true labels and masks.
          These are required in `validation_epoch_end` for feature concatenation.
        """
        del args, kwargs  # These variables are not used.

        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return Reverse Distillation trainer arguments."""
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS
