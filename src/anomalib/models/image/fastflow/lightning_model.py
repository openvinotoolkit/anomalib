"""FastFlow Lightning Model Implementation.

This module provides a PyTorch Lightning implementation of the FastFlow model for anomaly
detection. FastFlow is a fast flow-based model that uses normalizing flows to model the
distribution of features extracted from a pre-trained CNN backbone.

The model achieves competitive performance while maintaining fast inference times by
leveraging normalizing flows to transform feature distributions into a simpler form that
can be efficiently modeled.

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models import Fastflow
    >>> from anomalib.engine import Engine

    >>> datamodule = MVTecAD()
    >>> model = Fastflow()
    >>> engine = Engine()

    >>> engine.fit(model, datamodule=datamodule)  # doctest: +SKIP
    >>> predictions = engine.predict(model, datamodule=datamodule)  # doctest: +SKIP

Paper:
    Title: FastFlow: Unsupervised Anomaly Detection and Localization via 2D
           Normalizing Flows
    URL: https://arxiv.org/abs/2111.07677

See Also:
    :class:`anomalib.models.image.fastflow.torch_model.FastflowModel`:
        PyTorch implementation of the FastFlow model architecture.
    :class:`anomalib.models.image.fastflow.loss.FastflowLoss`:
        Loss function used to train the FastFlow model.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import AUROC, Evaluator, F1Score
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .loss import FastflowLoss
from .torch_model import FastflowModel


class Fastflow(AnomalibModule):
    """PL Lightning Module for the FastFlow algorithm.

    The FastFlow model uses normalizing flows to transform feature distributions from a
    pre-trained CNN backbone into a simpler form that can be efficiently modeled for
    anomaly detection.

    Args:
        backbone (str): Backbone CNN network architecture. Available options are
            ``"resnet18"``, ``"wide_resnet50_2"``, etc.
            Defaults to ``"resnet18"``.
        pre_trained (bool, optional): Whether to use pre-trained backbone weights.
            Defaults to ``True``.
        flow_steps (int, optional): Number of steps in the normalizing flow.
            Defaults to ``8``.
        conv3x3_only (bool, optional): Whether to use only 3x3 convolutions in the
            FastFlow model.
            Defaults to ``False``.
        hidden_ratio (float, optional): Ratio used to calculate hidden variable
            channels.
            Defaults to ``1.0``.
        pre_processor (PreProcessor | bool, optional): Pre-processor to use for
            input data.
            Defaults to ``True``.
        post_processor (PostProcessor | bool, optional): Post-processor to use for
            model outputs.
            Defaults to ``True``.
        evaluator (Evaluator | bool, optional): Evaluator to compute metrics.
            Defaults to ``True``.
        visualizer (Visualizer | bool, optional): Visualizer for model outputs.
            Defaults to ``True``.

    Raises:
        ValueError: If ``input_size`` is not provided during initialization.

    Example:
        >>> from anomalib.models import Fastflow
        >>> model = Fastflow(
        ...     backbone="resnet18",
        ...     pre_trained=True,
        ...     flow_steps=8
        ... )
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pre_trained: bool = True,
        flow_steps: int = 8,
        conv3x3_only: bool = False,
        hidden_ratio: float = 1.0,
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
            msg = "Fastflow needs input size to build torch model."
            raise ValueError(msg)

        self.backbone = backbone
        self.pre_trained = pre_trained
        self.flow_steps = flow_steps
        self.conv3x3_only = conv3x3_only
        self.hidden_ratio = hidden_ratio

        self.model = FastflowModel(
            input_size=self.input_size,
            backbone=self.backbone,
            pre_trained=self.pre_trained,
            flow_steps=self.flow_steps,
            conv3x3_only=self.conv3x3_only,
            hidden_ratio=self.hidden_ratio,
        )
        self.loss = FastflowLoss()

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform the training step input and return the loss.

        Args:
            batch (batch: dict[str, str | torch.Tensor]): Input batch
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            STEP_OUTPUT: Dictionary containing the loss value.
        """
        del args, kwargs  # These variables are not used.

        hidden_variables, jacobians = self.model(batch.image)
        loss = self.loss(hidden_variables, jacobians)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform the validation step and return the anomaly map.

        Args:
            batch (dict[str, str | torch.Tensor]): Input batch
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            STEP_OUTPUT | None: batch dictionary containing anomaly-maps.
        """
        del args, kwargs  # These variables are not used.

        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return FastFlow trainer arguments."""
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers for each decoder.

        Returns:
            Optimizer: Adam optimizer for each decoder
        """
        return optim.Adam(
            params=self.model.parameters(),
            lr=0.001,
            weight_decay=0.00001,
        )

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS

    @staticmethod
    def configure_evaluator() -> Evaluator:
        """Default evaluator.

        Override in subclass for model-specific evaluator behaviour.
        """
        # val metrics (needed for early stopping)
        image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
        pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
        val_metrics = [image_auroc, pixel_auroc]

        # test_metrics
        image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
        image_f1score = F1Score(fields=["pred_label", "gt_label"], prefix="image_")
        pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
        pixel_f1score = F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_")
        test_metrics = [image_auroc, image_f1score, pixel_auroc, pixel_f1score]
        return Evaluator(val_metrics=val_metrics, test_metrics=test_metrics)
