"""Student-Teacher Feature Pyramid Matching for anomaly detection.

This module implements the STFPM model for anomaly detection as described in
`Wang et al. (2021) <https://arxiv.org/abs/2103.04257>`_.

The model consists of:
- A pre-trained teacher network that extracts multi-scale features
- A student network that learns to match the teacher's feature representations
- Feature pyramid matching between student and teacher features
- Anomaly detection based on feature discrepancy

Example:
    >>> from anomalib.models.image import Stfpm
    >>> from anomalib.engine import Engine
    >>> from anomalib.data import MVTecAD
    >>> datamodule = MVTecAD()
    >>> model = Stfpm(
    ...     backbone="resnet18",
    ...     layers=["layer1", "layer2", "layer3"]
    ... )
    >>> engine = Engine(model=model, datamodule=datamodule)
    >>> engine.fit()  # doctest: +SKIP
    >>> predictions = engine.predict()  # doctest: +SKIP

See Also:
    - :class:`Stfpm`: Lightning implementation of the model
    - :class:`STFPMModel`: PyTorch implementation of the model architecture
    - :class:`STFPMLoss`: Loss function for training
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
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

from .loss import STFPMLoss
from .torch_model import STFPMModel

__all__ = ["Stfpm"]


class Stfpm(AnomalibModule):
    """PL Lightning Module for the STFPM algorithm.

    The Student-Teacher Feature Pyramid Matching (STFPM) model consists of a
    pre-trained teacher network and a student network that learns to match the
    teacher's feature representations. The model detects anomalies by comparing
    feature discrepancies between the teacher and student networks.

    Args:
        backbone (str): Name of the backbone CNN network used for both teacher
            and student. Defaults to ``"resnet18"``.
        layers (list[str]): Names of layers from which to extract features.
            Defaults to ``["layer1", "layer2", "layer3"]``.
        pre_processor (PreProcessor | bool, optional): Pre-processor to transform
            input data before passing to model. If ``True``, uses default.
            Defaults to ``True``.
        post_processor (PostProcessor | bool, optional): Post-processor to generate
            predictions from model outputs. If ``True``, uses default.
            Defaults to ``True``.
        evaluator (Evaluator | bool, optional): Evaluator to compute metrics.
            If ``True``, uses default. Defaults to ``True``.
        visualizer (Visualizer | bool, optional): Visualizer to display results.
            If ``True``, uses default. Defaults to ``True``.

    Example:
        >>> from anomalib.models.image import Stfpm
        >>> from anomalib.data import MVTecAD
        >>> from anomalib.engine import Engine
        >>> datamodule = MVTecAD()
        >>> model = Stfpm(
        ...     backbone="resnet18",
        ...     layers=["layer1", "layer2", "layer3"]
        ... )
        >>> engine = Engine(model=model, datamodule=datamodule)
        >>> engine.fit()  # doctest: +SKIP
        >>> predictions = engine.predict()  # doctest: +SKIP

    See Also:
        - :class:`anomalib.models.image.stfpm.torch_model.STFPMModel`:
            PyTorch implementation of the model architecture
        - :class:`anomalib.models.image.stfpm.loss.STFPMLoss`:
            Loss function for training
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: Sequence[str] = ("layer1", "layer2", "layer3"),
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

        self.model = STFPMModel(backbone=backbone, layers=layers)
        self.loss = STFPMLoss()

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform a training step of STFPM.

        For each batch, teacher and student features are extracted from the CNN.

        Args:
            batch (Batch): Input batch containing images and labels.
            args: Additional arguments (unused).
            kwargs: Additional keyword arguments (unused).

        Returns:
            STEP_OUTPUT: Dictionary containing the loss value.
        """
        del args, kwargs  # These variables are not used.

        teacher_features, student_features = self.model.forward(batch.image)
        loss = self.loss(teacher_features, student_features)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform a validation step of STFPM.

        Similar to training, extracts student/teacher features from CNN and
        computes anomaly maps.

        Args:
            batch (Batch): Input batch containing images and labels.
            args: Additional arguments (unused).
            kwargs: Additional keyword arguments (unused).

        Returns:
            STEP_OUTPUT: Dictionary containing images, anomaly maps, labels and
                masks for evaluation.
        """
        del args, kwargs  # These variables are not used.

        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Get required trainer arguments for the model.

        Returns:
            dict[str, Any]: Dictionary of trainer arguments:
                - ``gradient_clip_val``: Set to 0 to disable gradient clipping
                - ``num_sanity_val_steps``: Set to 0 to skip validation sanity
                  checks
        """
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers for training.

        Returns:
            torch.optim.Optimizer: SGD optimizer with the following parameters:
                - Learning rate: 0.4
                - Momentum: 0.9
                - Weight decay: 0.001
        """
        return optim.SGD(
            params=self.model.student_model.parameters(),
            lr=0.4,
            momentum=0.9,
            dampening=0.0,
            weight_decay=0.001,
        )

    @property
    def learning_type(self) -> LearningType:
        """Get the learning type of the model.

        Returns:
            LearningType: The model uses one-class learning.
        """
        return LearningType.ONE_CLASS
