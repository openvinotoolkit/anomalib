"""Pre-processing module for anomaly detection pipelines.

This module provides functionality for pre-processing data before model training
and inference through the :class:`PreProcessor` class.

The pre-processor handles:
    - Applying transforms to data during different pipeline stages
    - Managing stage-specific transforms (train/val/test)
    - Integrating with both PyTorch and Lightning workflows

Example:
    >>> from anomalib.pre_processing import PreProcessor
    >>> from torchvision.transforms.v2 import Resize
    >>> pre_processor = PreProcessor(transform=Resize(size=(256, 256)))
    >>> transformed_batch = pre_processor(batch)

The pre-processor is implemented as both a :class:`torch.nn.Module` and
:class:`lightning.pytorch.Callback` to support both inference and training
workflows.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.trainer.states import TrainerFn
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Transform

from .utils.transform import (
    get_dataloaders_transforms,
    get_exportable_transform,
    set_dataloaders_transforms,
    set_datamodule_stage_transform,
)

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

    from anomalib.data import AnomalibDataModule


class PreProcessor(nn.Module, Callback):
    """Anomalib pre-processor.

    This class serves as both a PyTorch module and a Lightning callback, handling
    the application of transforms to data batches during different stages of
    training, validation, testing, and prediction.

    Args:
        train_transform (Transform | None, optional): Transform to apply during
            training. Defaults to None.
        val_transform (Transform | None, optional): Transform to apply during
            validation. Defaults to None.
        test_transform (Transform | None, optional): Transform to apply during
            testing. Defaults to None.
        transform (Transform | None, optional): General transform to apply if
            stage-specific transforms are not provided. Defaults to None.

    Raises:
        ValueError: If both ``transform`` and any of the stage-specific transforms
            are provided simultaneously.

    Notes:
        If only ``transform`` is provided, it will be used for all stages (train,
        val, test).

        Priority of transforms:
            1. Explicitly set ``PreProcessor`` transforms (highest priority)
            2. Datamodule transforms (if ``PreProcessor`` has no transforms)
            3. Dataloader transforms (if neither ``PreProcessor`` nor datamodule
               have transforms)
            4. Default transforms (lowest priority)

    Example:
        >>> from torchvision.transforms.v2 import Compose, Resize, ToTensor
        >>> from anomalib.pre_processing import PreProcessor
        >>> # Define transforms
        >>> train_transform = Compose([
        ...     Resize((224, 224)),
        ...     ToTensor()
        ... ])
        >>> val_transform = Compose([
        ...     Resize((256, 256)),
        ...     CenterCrop((224, 224)),
        ...     ToTensor()
        ... ])
        >>> # Create PreProcessor with stage-specific transforms
        >>> pre_processor = PreProcessor(
        ...     train_transform=train_transform,
        ...     val_transform=val_transform
        ... )
        >>> # Create PreProcessor with a single transform for all stages
        >>> common_transform = Compose([
        ...     Resize((224, 224)),
        ...     ToTensor()
        ... ])
        >>> pre_processor_common = PreProcessor(transform=common_transform)

    Integration with Lightning:
        >>> class MyModel(LightningModule):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.pre_processor = PreProcessor(...)
        ...
        ...     def configure_callbacks(self):
        ...         return [self.pre_processor]
        ...
        ...     def training_step(self, batch, batch_idx):
        ...         # Pre-processor automatically applies correct transform
        ...         processed_batch = self.pre_processor(batch)
        ...         # Rest of the training step
    """

    def __init__(
        self,
        train_transform: Transform | None = None,
        val_transform: Transform | None = None,
        test_transform: Transform | None = None,
        transform: Transform | None = None,
    ) -> None:
        super().__init__()

        if transform and any([train_transform, val_transform, test_transform]):
            msg = (
                "`transforms` cannot be used together with `train_transform`, `val_transform`, `test_transform`.\n"
                "If you want to apply the same transform to the training, validation and test data, "
                "use only `transforms`. \n"
                "Otherwise, specify transforms for training, validation and test individually."
            )
            raise ValueError(msg)

        self.train_transform = train_transform or transform
        self.val_transform = val_transform or transform
        self.test_transform = test_transform or transform
        self.predict_transform = self.test_transform
        self.export_transform = get_exportable_transform(self.test_transform)

    def setup_datamodule_transforms(self, datamodule: "AnomalibDataModule") -> None:
        """Set up datamodule transforms.

        Args:
            datamodule (AnomalibDataModule): The datamodule to configure
                transforms for.
        """
        # If PreProcessor has transforms, propagate them to datamodule
        if any([self.train_transform, self.val_transform, self.test_transform]):
            transforms = {
                "fit": self.train_transform,
                "val": self.val_transform,
                "test": self.test_transform,
                "predict": self.predict_transform,
            }

            for stage, transform in transforms.items():
                if transform is not None:
                    set_datamodule_stage_transform(datamodule, transform, stage)

    def setup_dataloader_transforms(self, dataloaders: "EVAL_DATALOADERS | TRAIN_DATALOADERS") -> None:
        """Set up dataloader transforms.

        Args:
            dataloaders (EVAL_DATALOADERS | TRAIN_DATALOADERS): The dataloaders
                to configure transforms for.
        """
        if isinstance(dataloaders, DataLoader):
            dataloaders = [dataloaders]

        # If PreProcessor has transforms, propagate them to dataloaders
        if any([self.train_transform, self.val_transform, self.test_transform]):
            transforms = {
                "train": self.train_transform,
                "val": self.val_transform,
                "test": self.test_transform,
            }
            set_dataloaders_transforms(dataloaders, transforms)
            return

        # Try to get transforms from dataloaders
        if dataloaders:
            dataloaders_transforms = get_dataloaders_transforms(dataloaders)
            if dataloaders_transforms:
                self.train_transform = dataloaders_transforms.get("train")
                self.val_transform = dataloaders_transforms.get("val")
                self.test_transform = dataloaders_transforms.get("test")
                self.predict_transform = self.test_transform
                self.export_transform = get_exportable_transform(self.test_transform)

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Configure transforms at the start of each stage.

        Args:
            trainer (Trainer): The Lightning trainer.
            pl_module (LightningModule): The Lightning module.
            stage (str): The stage (e.g., 'fit', 'validate', 'test', 'predict').
        """
        stage = TrainerFn(stage).value  # Ensure stage is str

        if hasattr(trainer, "datamodule"):
            self.setup_datamodule_transforms(datamodule=trainer.datamodule)
        elif hasattr(trainer, f"{stage}_dataloaders"):
            dataloaders = getattr(trainer, f"{stage}_dataloaders")
            self.setup_dataloader_transforms(dataloaders=dataloaders)

        super().setup(trainer, pl_module, stage)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Apply transforms to the batch of tensors for inference.

        This forward-pass is only used after the model is exported.
        Within the Lightning training/validation/testing loops, the transforms are
        applied in the ``on_*_batch_start`` methods.

        Args:
            batch (torch.Tensor): Input batch to transform.

        Returns:
            torch.Tensor: Transformed batch.
        """
        return self.export_transform(batch) if self.export_transform else batch
