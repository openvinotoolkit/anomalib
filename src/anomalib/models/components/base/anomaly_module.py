"""Base Anomaly Module for Training Task."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightning.pytorch as pl
import torch
from lightning.pytorch import Callback
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torchvision.transforms.v2 import Compose, Normalize, Resize

from anomalib import LearningType
from anomalib.data import Batch, InferenceBatch
from anomalib.metrics.threshold import Threshold
from anomalib.post_processing import OneClassPostProcessor, PostProcessor
from anomalib.pre_processing import PreProcessor

from .export_mixin import ExportMixin

if TYPE_CHECKING:
    from anomalib.metrics import AnomalibMetricCollection

logger = logging.getLogger(__name__)


class AnomalyModule(ExportMixin, pl.LightningModule, ABC):
    """AnomalyModule to train, validate, predict and test images.

    Acts as a base class for all the Anomaly Modules in the library.
    """

    def __init__(
        self,
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | None = None,
    ) -> None:
        super().__init__()
        logger.info("Initializing %s model.", self.__class__.__name__)

        self.save_hyperparameters()
        self.model: nn.Module
        self.loss: nn.Module
        self.callbacks: list[Callback]

        self.image_metrics: AnomalibMetricCollection
        self.pixel_metrics: AnomalibMetricCollection

        # Handle pre-processor
        # True -> use default pre-processor
        # False -> no pre-processor
        # PreProcessor -> use the provided pre-processor
        if isinstance(pre_processor, PreProcessor):
            self.pre_processor = pre_processor
        elif isinstance(pre_processor, bool):
            self.pre_processor = self.configure_pre_processor()
        else:
            msg = f"Invalid pre-processor type: {type(pre_processor)}"
            raise TypeError(msg)

        self.post_processor = post_processor or self.default_post_processor()

        self._input_size: tuple[int, int] | None = None

        self._is_setup = False  # flag to track if setup has been called from the trainer

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

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Called when loading a checkpoint.

        This method is called to ensure that the `TorchModel` is built before
        loading the state dict.
        """
        del checkpoint  # `checkpoint` variable is not used.
        self.setup(stage="load_checkpoint")

    def configure_callbacks(self) -> Sequence[Callback] | Callback:
        """Configure default callbacks for AnomalyModule."""
        return [self.pre_processor]

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

    def _save_to_state_dict(self, destination: OrderedDict, prefix: str, keep_vars: bool) -> None:
        if hasattr(self, "image_threshold"):
            destination["image_threshold_class"] = (
                f"{self.image_threshold.__class__.__module__}.{self.image_threshold.__class__.__name__}"
            )
        if hasattr(self, "pixel_threshold"):
            destination["pixel_threshold_class"] = (
                f"{self.pixel_threshold.__class__.__module__}.{self.pixel_threshold.__class__.__name__}"
            )
        if hasattr(self, "normalization_metrics"):
            for metric in self.normalization_metrics:
                metric_class = self.normalization_metrics[metric].__class__
                destination[f"{metric}_normalization_class"] = f"{metric_class.__module__}.{metric_class.__name__}"

        return super()._save_to_state_dict(destination, prefix, keep_vars)

    @staticmethod
    def _get_instance(state_dict: OrderedDict[str, Any], dict_key: str) -> Threshold:
        """Get the threshold class from the ``state_dict``."""
        class_path = state_dict.pop(dict_key)
        module = importlib.import_module(".".join(class_path.split(".")[:-1]))
        return getattr(module, class_path.split(".")[-1])()

    @property
    @abstractmethod
    def learning_type(self) -> LearningType:
        """Learning type of the model."""
        raise NotImplementedError

    @classmethod
    def configure_pre_processor(
        cls,
        image_size: tuple[int, int] | None = None,
        **kwargs: Any,  # noqa: ANN401, ARG003
    ) -> PreProcessor:
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

            >>> preprocessor = AnomalyModule.configure_pre_processor(image_size=(512, 512))

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

    def default_post_processor(self) -> PostProcessor:
        """Default post processor.

        Override in subclass for model-specific post-processing behaviour.
        """
        if self.learning_type == LearningType.ONE_CLASS:
            return OneClassPostProcessor()
        msg = f"No default post-processor available for model {self.__name__} with learning type {self.learning_type}. \
              Please override the default_post_processor method in the model implementation."
        raise NotImplementedError(msg)

    @property
    def input_size(self) -> tuple[int, int] | None:
        """Return the effective input size of the model.

        The effective input size is the size of the input tensor after the transform has been applied. If the transform
        is not set, or if the transform does not change the shape of the input tensor, this method will return None.
        """
        transform = self.pre_processor.train_transform
        if transform is None:
            return None
        dummy_input = torch.zeros(1, 3, 1, 1)
        output_shape = transform(dummy_input).shape[-2:]
        return None if output_shape == (1, 1) else output_shape[-2:]

    @classmethod
    def from_config(
        cls: type["AnomalyModule"],
        config_path: str | Path,
        **kwargs,
    ) -> "AnomalyModule":
        """Create a model instance from the configuration.

        Args:
            config_path (str | Path): Path to the model configuration file.
            **kwargs (dict): Additional keyword arguments.

        Returns:
            AnomalyModule: model instance.

        Example:
            The following example shows how to get model from patchcore.yaml:

            .. code-block:: python
                >>> model_config = "configs/model/patchcore.yaml"
                >>> model = AnomalyModule.from_config(config_path=model_config)

            The following example shows overriding the configuration file with additional keyword arguments:

            .. code-block:: python
                >>> override_kwargs = {"model.pre_trained": False}
                >>> model = AnomalyModule.from_config(config_path=model_config, **override_kwargs)
        """
        from jsonargparse import ActionConfigFile, ArgumentParser
        from lightning.pytorch import Trainer

        from anomalib import TaskType

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
        model_parser.add_subclass_arguments(AnomalyModule, "model", required=False, fail_untyped=False)
        model_parser.add_argument("--task", type=TaskType | str, default=TaskType.SEGMENTATION)
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
        if isinstance(model, AnomalyModule):
            return model

        msg = f"Model is not an instance of AnomalyModule: {model}"
        raise ValueError(msg)
