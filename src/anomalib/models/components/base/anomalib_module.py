"""Base Anomaly Module for Training Task.

This module provides the foundational class for all anomaly detection models in
anomalib. The ``AnomalibModule`` class extends PyTorch Lightning's
``LightningModule`` and provides common functionality for training, validation,
testing and inference of anomaly detection models.

The class handles:
- Model initialization and setup
- Pre-processing of input data
- Post-processing of model outputs
- Evaluation metrics computation
- Visualization of results
- Model export capabilities

Example:
    Create a custom anomaly detection model:

    >>> from anomalib.models.components.base import AnomalibModule
    >>> class MyModel(AnomalibModule):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.model = torch.nn.Linear(10, 1)
    ...
    ...     def training_step(self, batch, batch_idx):
    ...         return self.model(batch)

    Create model with custom components:

    >>> from anomalib.pre_processing import PreProcessor
    >>> from anomalib.post_processing import PostProcessor
    >>> model = MyModel(
    ...     pre_processor=PreProcessor(),
    ...     post_processor=PostProcessor(),
    ...     evaluator=True,
    ...     visualizer=True
    ... )
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torchvision.transforms.v2 import Compose, Normalize, Resize

from anomalib import LearningType
from anomalib.data import Batch, InferenceBatch
from anomalib.metrics import AUROC, F1Score
from anomalib.metrics.evaluator import Evaluator
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import ImageVisualizer, Visualizer

from .export_mixin import ExportMixin

logger = logging.getLogger(__name__)


class AnomalibModule(ExportMixin, pl.LightningModule, ABC):
    """Base class for all anomaly detection modules in anomalib.

    This class provides the core functionality for training, validation, testing
    and inference of anomaly detection models. It handles data pre-processing,
    post-processing, evaluation and visualization.

    Args:
        pre_processor (PreProcessor | bool, optional): Pre-processor instance or
            flag to use default. Defaults to ``True``.
        post_processor (PostProcessor | bool, optional): Post-processor instance
            or flag to use default. Defaults to ``True``.
        evaluator (Evaluator | bool, optional): Evaluator instance or flag to use
            default. Defaults to ``True``.
        visualizer (Visualizer | bool, optional): Visualizer instance or flag to
            use default. Defaults to ``True``.

    Attributes:
        model (nn.Module): PyTorch model to be trained
        loss (nn.Module): Loss function for training
        callbacks (list[Callback]): List of callbacks
        pre_processor (PreProcessor | None): Component for pre-processing inputs
        post_processor (PostProcessor | None): Component for post-processing
            outputs
        evaluator (Evaluator | None): Component for computing metrics
        visualizer (Visualizer | None): Component for visualization

    Example:
        Create a model with default components:

        >>> model = AnomalibModule()

        Create a model with custom components:

        >>> from anomalib.pre_processing import PreProcessor
        >>> from anomalib.post_processing import PostProcessor
        >>> model = AnomalibModule(
        ...     pre_processor=PreProcessor(),
        ...     post_processor=PostProcessor(),
        ...     evaluator=True,
        ...     visualizer=True
        ... )

        Disable certain components:

        >>> model = AnomalibModule(
        ...     pre_processor=False,
        ...     post_processor=False,
        ...     evaluator=False,
        ...     visualizer=False
        ... )
    """

    def __init__(
        self,
        pre_processor: nn.Module | bool = True,
        post_processor: nn.Module | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__()
        logger.info("Initializing %s model.", self.__class__.__name__)

        self.save_hyperparameters()
        self.model: nn.Module
        self.loss: nn.Module
        self.callbacks: list[Callback]

        self.pre_processor = self._resolve_component(pre_processor, nn.Module, self.configure_pre_processor)
        self.post_processor = self._resolve_component(post_processor, nn.Module, self.configure_post_processor)
        self.evaluator = self._resolve_component(evaluator, Evaluator, self.configure_evaluator)
        self.visualizer = self._resolve_component(visualizer, Visualizer, self.configure_visualizer)

        self._input_size: tuple[int, int] | None = None

    @property
    def name(self) -> str:
        """Get name of the model.

        Returns:
            str: Name of the model class
        """
        return self.__class__.__name__

    def configure_callbacks(self) -> Sequence[Callback] | Callback:
        """Configure callbacks for the model.

        Returns:
            Sequence[Callback] | Callback: List of callbacks including components
                that inherit from ``Callback``

        Example:
            >>> model = AnomalibModule()
            >>> callbacks = model.configure_callbacks()
            >>> isinstance(callbacks, (Sequence, Callback))
            True
        """
        callbacks: list[Callback] = []
        callbacks.extend(
            component
            for component in (self.pre_processor, self.post_processor, self.evaluator, self.visualizer)
            if isinstance(component, Callback)
        )
        return callbacks

    def forward(self, batch: torch.Tensor, *args, **kwargs) -> InferenceBatch:
        """Perform forward pass through the model pipeline.

        The input batch is passed through:
        1. Pre-processor (if configured)
        2. Model
        3. Post-processor (if configured)

        Args:
            batch (torch.Tensor): Input batch
            *args: Additional positional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            InferenceBatch: Processed batch with model predictions

        Example:
            >>> model = AnomalibModule()
            >>> batch = torch.randn(1, 3, 256, 256)
            >>> output = model(batch)
            >>> isinstance(output, InferenceBatch)
            True
        """
        del args, kwargs  # These variables are not used.
        batch = self.pre_processor(batch) if self.pre_processor else batch
        batch = self.model(batch)
        return self.post_processor(batch) if self.post_processor else batch

    def predict_step(
        self,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> STEP_OUTPUT:
        """Perform prediction step.

        This method is called during the predict stage of training. By default,
        it calls the validation step.

        Args:
            batch (Batch): Input batch
            batch_idx (int): Index of the batch
            dataloader_idx (int, optional): Index of the dataloader.
                Defaults to ``0``.

        Returns:
            STEP_OUTPUT: Model predictions
        """
        del dataloader_idx  # These variables are not used.

        return self.validation_step(batch, batch_idx)

    def test_step(self, batch: Batch, batch_idx: int, *args, **kwargs) -> STEP_OUTPUT:
        """Perform test step.

        This method is called during the test stage of training. By default,
        it calls the predict step.

        Args:
            batch (Batch): Input batch
            batch_idx (int): Index of the batch
            *args: Additional positional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            STEP_OUTPUT: Model predictions
        """
        del args, kwargs  # These variables are not used.

        return self.predict_step(batch, batch_idx)

    @property
    @abstractmethod
    def trainer_arguments(self) -> dict[str, Any]:
        """Get trainer arguments specific to this model.

        Returns:
            dict[str, Any]: Dictionary of trainer arguments

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def learning_type(self) -> LearningType:
        """Get learning type of the model.

        Returns:
            LearningType: Type of learning (e.g. one-class, supervised)

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError

    @staticmethod
    def _resolve_component(
        component: nn.Module | None,
        component_type: type,
        default_callable: Callable,
    ) -> nn.Module | None:
        """Resolve and validate the subcomponent configuration.

        This method resolves the configuration for various subcomponents like
        pre-processor, post-processor, evaluator and visualizer. It validates
        the configuration and returns the configured component. If the component
        is a boolean, it uses the default callable to create the component. If
        the component is already an instance of the component type, it returns
        the component as is.

        Args:
            component (object): Component configuration
            component_type (Type): Type of the component
            default_callable (Callable): Callable to create default component

        Returns:
            Component | None: Configured component

        Raises:
            TypeError: If component is invalid type
        """
        if isinstance(component, component_type):
            return component
        if isinstance(component, bool):
            return default_callable() if component else None
        msg = f"Passed object should be {component_type} or bool, got: {type(component)}"
        raise TypeError(msg)

    @staticmethod
    def configure_pre_processor(image_size: tuple[int, int] | None = None) -> PreProcessor:
        """Configure the default pre-processor.

        The default pre-processor resizes images and normalizes using ImageNet
        statistics. Override this method to provide a custom pre-processor for
        the model.

        Args:
            image_size (tuple[int, int] | None, optional): Target size for
                resizing. Defaults to ``(256, 256)``.

        Returns:
            PreProcessor: Configured pre-processor

        Example:
            >>> preprocessor = AnomalibModule.configure_pre_processor((512, 512))
            >>> isinstance(preprocessor, PreProcessor)
            True
        """
        image_size = image_size or (256, 256)
        return PreProcessor(
            transform=Compose([
                Resize(image_size, antialias=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        )

    def configure_post_processor(self) -> PostProcessor | None:
        """Configure the default post-processor.

        The default post-processor is based on the model's learning type. Override
        this method to provide a custom post-processor for the model.

        Returns:
            PostProcessor | None: Configured post-processor based on learning type

        Raises:
            NotImplementedError: If no default post-processor exists for the
                model's learning type

        Example:
            >>> model = AnomalibModule()
            >>> post_processor = model.configure_post_processor()
            >>> isinstance(post_processor, PostProcessor)
            True
        """
        if self.learning_type == LearningType.ONE_CLASS:
            return PostProcessor()
        msg = (
            f"No default post-processor available for model with learning type "
            f"{self.learning_type}. Please override configure_post_processor."
        )
        raise NotImplementedError(msg)

    @staticmethod
    def configure_evaluator() -> Evaluator:
        """Configure the default evaluator.

        The default evaluator includes metrics for both image-level and
        pixel-level evaluation. Override this method to provide custom metrics for the model.

        Returns:
            Evaluator: Configured evaluator with default metrics

        Example:
            >>> evaluator = AnomalibModule.configure_evaluator()
            >>> isinstance(evaluator, Evaluator)
            True
        """
        image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
        image_f1score = F1Score(fields=["pred_label", "gt_label"], prefix="image_")
        pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False)
        pixel_f1score = F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_", strict=False)
        test_metrics = [image_auroc, image_f1score, pixel_auroc, pixel_f1score]
        return Evaluator(test_metrics=test_metrics)

    @classmethod
    def configure_visualizer(cls) -> ImageVisualizer:
        """Configure the default visualizer.

        Override this method to provide a custom visualizer for the model.

        Returns:
            ImageVisualizer: Default image visualizer instance

        Example:
            >>> visualizer = AnomalibModule.configure_visualizer()
            >>> isinstance(visualizer, ImageVisualizer)
            True
        """
        return ImageVisualizer()

    @property
    def input_size(self) -> tuple[int, int] | None:
        """Get the effective input size of the model.

        Returns:
            tuple[int, int] | None: Height and width of model input after
                pre-processing, or ``None`` if size cannot be determined

        Example:
            >>> model = AnomalibModule()
            >>> model.input_size  # Returns size after pre-processing
            (256, 256)
        """
        transform = self.pre_processor.transform if self.pre_processor else None
        if transform is None:
            return None
        dummy_input = torch.zeros(1, 3, 1, 1)
        output_shape = transform(dummy_input).shape[-2:]
        return None if output_shape == (1, 1) else output_shape[-2:]

    @classmethod
    def from_config(
        cls: type["AnomalibModule"],
        config_path: str | Path,
        **kwargs,
    ) -> "AnomalibModule":
        """Create a model instance from a configuration file.

        Args:
            config_path (str | Path): Path to the model configuration file
            **kwargs: Additional arguments to override config values

        Returns:
            AnomalibModule: Instantiated model

        Raises:
            FileNotFoundError: If config file does not exist
            ValueError: If instantiated model is not AnomalibModule

        Example:
            >>> model = AnomalibModule.from_config("examples/configs/model/patchcore.yaml")
            >>> isinstance(model, AnomalibModule)
            True

            Override config values:
            >>> model = AnomalibModule.from_config(
            ...     "examples/configs/model/patchcore.yaml",
            ...     model__backbone="resnet18"
            ... )
        """
        from jsonargparse import ActionConfigFile, ArgumentParser
        from lightning.pytorch import Trainer

        if not Path(config_path).exists():
            msg = f"Configuration file not found: {config_path}"
            raise FileNotFoundError(msg)

        model_parser = ArgumentParser()
        model_parser.add_argument(
            "-c",
            "--config",
            action=ActionConfigFile,
            help="Path to a configuration file in json or yaml format.",
        )
        model_parser.add_subclass_arguments(AnomalibModule, "model", required=False, fail_untyped=False)
        model_parser.add_class_arguments(Trainer, "trainer", fail_untyped=False, instantiate=False, sub_configs=True)
        args = ["--config", str(config_path)]
        for key, value in kwargs.items():
            args.extend([f"--{key}", str(value)])
        config = model_parser.parse_args(args=args)
        instantiated_classes = model_parser.instantiate_classes(config)
        model = instantiated_classes.get("model")
        if isinstance(model, AnomalibModule):
            return model

        msg = f"Model is not an instance of AnomalibModule: {model}"
        raise ValueError(msg)


class AnomalyModule(AnomalibModule):
    """Deprecated AnomalyModule class. Use AnomalibModule instead."""

    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "AnomalyModule is deprecated and will be removed in a future release. Use AnomalibModule instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
