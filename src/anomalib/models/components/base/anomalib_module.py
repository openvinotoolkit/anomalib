"""Base Anomaly Module for Training Task."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch import Callback
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torchvision.transforms.v2 import Compose, Normalize, Resize

from anomalib import LearningType
from anomalib.data import Batch, InferenceBatch
from anomalib.metrics import AUROC, F1Score
from anomalib.metrics.evaluator import Evaluator
from anomalib.metrics.threshold import Threshold
from anomalib.post_processing import OneClassPostProcessor, PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import ImageVisualizer, Visualizer

from .export_mixin import ExportMixin

logger = logging.getLogger(__name__)


class AnomalibModule(ExportMixin, pl.LightningModule, ABC):
    """AnomalibModule to train, validate, predict and test images.

    Acts as a base class for all the Anomaly Modules in the library.
    """

    def __init__(
        self,
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__()
        logger.info("Initializing %s model.", self.__class__.__name__)

        self.save_hyperparameters()
        self.model: nn.Module
        self.loss: nn.Module
        self.callbacks: list[Callback]

        self.pre_processor = self._resolve_pre_processor(pre_processor)
        self.post_processor = self._resolve_post_processor(post_processor)
        self.evaluator = self._resolve_evaluator(evaluator)
        self.visualizer = self._resolve_visualizer(visualizer)

        self._input_size: tuple[int, int] | None = None
        self._is_setup = False

    @property
    def name(self) -> str:
        """Name of the model."""
        return self.__class__.__name__

    def setup(self, stage: str | None = None) -> None:
        """Calls the _setup method to build the model if the model is not already built."""
        if getattr(self, "model", None) is None or not self._is_setup:
            self._setup()
            if isinstance(stage, TrainerFn):
                # only set the flag if the stage is a TrainerFn, which means the setup has been called from a trainer
                self._is_setup = True

    def _setup(self) -> None:
        """The _setup method is used to build the torch model dynamically or adjust something about them.

        The model implementer may override this method to build the model. This is useful when the model cannot be set
        in the `__init__` method because it requires some information or data that is not available at the time of
        initialization.
        """

    def configure_callbacks(self) -> Sequence[Callback] | Callback:
        """Configure default callbacks for AnomalibModule.

        Returns:
            List of callbacks that includes the pre-processor, post-processor, evaluator,
            and visualizer if they are available and inherit from Callback.
        """
        callbacks: list[Callback] = []
        callbacks.extend(
            component
            for component in (self.pre_processor, self.post_processor, self.evaluator, self.visualizer)
            if isinstance(component, Callback)
        )
        return callbacks

    def forward(self, batch: torch.Tensor, *args, **kwargs) -> InferenceBatch:
        """Perform the forward-pass by passing input tensor to the module.

        Args:
            batch (dict[str, str | torch.Tensor]): Input batch.
            *args: Arguments.
            **kwargs: Keyword arguments.

        Returns:
            Tensor: Output tensor from the model.
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
        """Step function called during :meth:`~lightning.pytorch.trainer.Trainer.predict`.

        By default, it calls :meth:`~lightning.pytorch.core.lightning.LightningModule.forward`.
        Override to add any processing logic.

        Args:
            batch (Any): Current batch
            batch_idx (int): Index of current batch
            dataloader_idx (int): Index of the current dataloader

        Return:
            Predicted output
        """
        del dataloader_idx  # These variables are not used.

        return self.validation_step(batch, batch_idx)

    def test_step(self, batch: Batch, batch_idx: int, *args, **kwargs) -> STEP_OUTPUT:
        """Calls validation_step for anomaly map/score calculation.

        Args:
          batch (Batch): Input batch
          batch_idx (int): Batch index
          args: Arguments.
          kwargs: Keyword arguments.

        Returns:
          Dictionary containing images, features, true labels and masks.
          These are required in `validation_epoch_end` for feature concatenation.
        """
        del args, kwargs  # These variables are not used.

        return self.predict_step(batch, batch_idx)

    @property
    @abstractmethod
    def trainer_arguments(self) -> dict[str, Any]:
        """Arguments used to override the trainer parameters so as to train the model correctly."""
        raise NotImplementedError

    @property
    @abstractmethod
    def learning_type(self) -> LearningType:
        """Learning type of the model."""
        raise NotImplementedError

    def _resolve_pre_processor(self, pre_processor: PreProcessor | bool) -> PreProcessor | None:
        """Resolve and validate which pre-processor to use..

        Args:
            pre_processor: Pre-processor configuration
                - True -> use default pre-processor
                - False -> no pre-processor
                - PreProcessor -> use the provided pre-processor

        Returns:
            Configured pre-processor
        """
        if isinstance(pre_processor, PreProcessor):
            return pre_processor
        if isinstance(pre_processor, bool):
            return self.configure_pre_processor() if pre_processor else None
        msg = f"Invalid pre-processor type: {type(pre_processor)}"
        raise TypeError(msg)

    @classmethod
    def configure_pre_processor(cls, image_size: tuple[int, int] | None = None) -> PreProcessor:
        """Configure the pre-processor.

        The default pre-processor resizes images to 256x256 and normalizes using ImageNet statistics.
        Individual models can override this method to provide custom transforms and pre-processing pipelines.

        Args:
            image_size (tuple[int, int] | None, optional): Target size for resizing images.
                If None, defaults to (256, 256). Defaults to None.
            **kwargs (Any): Additional keyword arguments (unused).

        Returns:
            PreProcessor: Configured pre-processor instance.

        Examples:
            Get default pre-processor with custom image size:

            >>> preprocessor = AnomalibModule.configure_pre_processor(image_size=(512, 512))

            Create model with custom pre-processor:

            >>> from torchvision.transforms.v2 import RandomHorizontalFlip
            >>> custom_transform = Compose([
            ...     Resize((256, 256), antialias=True),
            ...     CenterCrop((224, 224)),
            ...     RandomHorizontalFlip(p=0.5),
            ...     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ... ])
            >>> preprocessor.train_transform = custom_transform
            >>> model = PatchCore(pre_processor=preprocessor)

            Disable pre-processing:

            >>> model = PatchCore(pre_processor=False)
        """
        image_size = image_size or (256, 256)
        return PreProcessor(
            transform=Compose([
                Resize(image_size, antialias=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        )

    def _resolve_post_processor(self, post_processor: PostProcessor | bool) -> PostProcessor | None:
        """Resolve and validate which post-processor to use.

        Args:
            post_processor: Post-processor configuration
                - True -> use default post-processor
                - False -> no post-processor
                - PostProcessor -> use the provided post-processor

        Returns:
            Configured post-processor
        """
        if isinstance(post_processor, PostProcessor):
            return post_processor
        if isinstance(post_processor, bool):
            return self.configure_post_processor() if post_processor else None
        msg = f"Invalid post-processor type: {type(post_processor)}"
        raise TypeError(msg)

    def configure_post_processor(self) -> PostProcessor | None:
        """Configure the default post-processor based on the learning type.

        Returns:
            PostProcessor: Configured post-processor instance.

        Raises:
            NotImplementedError: If no default post-processor is available for the model's learning type.

        Examples:
            Get default post-processor:

            >>> post_processor = AnomalibModule.configure_post_processor()

            Create model with custom post-processor:

            >>> custom_post_processor = CustomPostProcessor()
            >>> model = PatchCore(post_processor=custom_post_processor)

            Disable post-processing:

            >>> model = PatchCore(post_processor=False)
        """
        if self.learning_type == LearningType.ONE_CLASS:
            return OneClassPostProcessor()
        msg = (
            f"No default post-processor available for model with learning type {self.learning_type}. "
            "Please override the configure_post_processor method in the model implementation."
        )
        raise NotImplementedError(msg)

    def _resolve_evaluator(self, evaluator: Evaluator | bool) -> Evaluator | None:
        """Resolve the evaluator to be used in the model.

        If the evaluator is set to True, the default evaluator will be used. If the evaluator is set to False, no
        evaluator will be used. If the evaluator is an instance of Evaluator, it will be used as the evaluator.
        """
        if isinstance(evaluator, Evaluator):
            return evaluator
        if isinstance(evaluator, bool):
            return self.configure_evaluator() if evaluator else None
        msg = f"evaluator must be of type Evaluator or bool, got {type(evaluator)}"
        raise TypeError(msg)

    @staticmethod
    def configure_evaluator() -> Evaluator:
        """Default evaluator.

        Override in subclass for model-specific evaluator behaviour.
        """
        image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
        image_f1score = F1Score(fields=["pred_label", "gt_label"], prefix="image_")
        pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
        pixel_f1score = F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_")
        test_metrics = [image_auroc, image_f1score, pixel_auroc, pixel_f1score]
        return Evaluator(test_metrics=test_metrics)

    def _resolve_visualizer(self, visualizer: Visualizer | bool) -> Visualizer | None:
        """Resolve and validate which visualizer to use.

        Args:
            visualizer: Visualizer configuration
                - True -> use default visualizer
                - False -> no visualizer
                - Visualizer -> use the provided visualizer

        Returns:
            Configured visualizer
        """
        if isinstance(visualizer, Visualizer):
            return visualizer
        if isinstance(visualizer, bool):
            return self.configure_visualizer() if visualizer else None
        msg = f"Visualizer must be of type Visualizer or bool, got {type(visualizer)}"
        raise TypeError(msg)

    @classmethod
    def configure_visualizer(cls) -> ImageVisualizer:
        """Configure the default visualizer.

        By default, this method returns an ImageVisualizer instance, which is suitable for
        visualizing image-based anomaly detection results. However, the visualizer can be
        customized based on your needs - for example, using VideoVisualizer for video data
        or implementing a custom visualizer for specific visualization requirements.

        Returns:
            Visualizer: Configured visualizer instance (ImageVisualizer by default).

        Examples:
            Get default ImageVisualizer:

            >>> visualizer = AnomalibModule.configure_visualizer()

            Create model with VideoVisualizer:

            >>> from custom_module import VideoVisualizer
            >>> video_visualizer = VideoVisualizer()
            >>> model = PatchCore(visualizer=video_visualizer)

            Create model with custom visualizer:

            >>> class CustomVisualizer(Visualizer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         # Custom visualization logic
            >>> custom_visualizer = CustomVisualizer()
            >>> model = PatchCore(visualizer=custom_visualizer)

            Disable visualization:

            >>> model = PatchCore(visualizer=False)
        """
        return ImageVisualizer()

    @property
    def input_size(self) -> tuple[int, int] | None:
        """Return the effective input size of the model.

        The effective input size is the size of the input tensor after the transform has been applied. If the transform
        is not set, or if the transform does not change the shape of the input tensor, this method will return None.
        """
        transform = self.pre_processor.predict_transform if self.pre_processor else None
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
        """Create a model instance from the configuration.

        Args:
            config_path (str | Path): Path to the model configuration file.
            **kwargs (dict): Additional keyword arguments.

        Returns:
            AnomalibModule: model instance.

        Example:
            The following example shows how to get model from patchcore.yaml:

            .. code-block:: python
                >>> model_config = "configs/model/patchcore.yaml"
                >>> model = AnomalibModule.from_config(config_path=model_config)

            The following example shows overriding the configuration file with additional keyword arguments:

            .. code-block:: python
                >>> override_kwargs = {"model.pre_trained": False}
                >>> model = AnomalibModule.from_config(config_path=model_config, **override_kwargs)
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
        model_parser.add_argument("--metrics.image", type=list[str] | str | None, default=["F1Score", "AUROC"])
        model_parser.add_argument("--metrics.pixel", type=list[str] | str | None, default=None, required=False)
        model_parser.add_argument("--metrics.threshold", type=Threshold | str, default="F1AdaptiveThreshold")
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
