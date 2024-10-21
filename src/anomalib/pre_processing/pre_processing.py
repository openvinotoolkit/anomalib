"""Anomalib pre-processing module."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.trainer.states import TrainerFn
from torch import nn
from torchvision.transforms.v2 import Transform

from anomalib.data.dataclasses.torch.base import Batch
from anomalib.data.utils.transform import set_dataloader_transform, set_datamodule_transform
from anomalib.deploy.utils import get_exportable_transform


class PreProcessor(nn.Module, Callback):
    """Anomalib pre-processor.

    This class serves as both a PyTorch module and a Lightning callback, handling
    the application of transforms to data batches during different stages of
    training, validation, testing, and prediction.

    Args:
        train_transform (Transform | None): Transform to apply during training.
        val_transform (Transform | None): Transform to apply during validation.
        test_transform (Transform | None): Transform to apply during testing.
        transform (Transform | None): General transform to apply if stage-specific
            transforms are not provided.

    Raises:
        ValueError: If both `transform` and any of the stage-specific transforms
            are provided simultaneously.

    Notes:
        If only `transform` is provided, it will be used for all stages (train, val, test).

    Examples:
        >>> from torchvision.transforms.v2 import Compose, Resize, ToTensor
        >>> from anomalib.pre_processing import PreProcessor

        >>> # Define transforms
        >>> train_transform = Compose([Resize((224, 224)), ToTensor()])
        >>> val_transform = Compose([Resize((256, 256)), CenterCrop((224, 224)), ToTensor()])

        >>> # Create PreProcessor with stage-specific transforms
        >>> pre_processor = PreProcessor(
        ...     train_transform=train_transform,
        ...     val_transform=val_transform
        ... )

        >>> # Create PreProcessor with a single transform for all stages
        >>> common_transform = Compose([Resize((224, 224)), ToTensor()])
        >>> pre_processor_common = PreProcessor(transform=common_transform)

        >>> # Use in a Lightning module
        >>> class MyModel(LightningModule):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.pre_processor = PreProcessor(...)
        ...
        ...     def configure_callbacks(self):
        ...         return [self.pre_processor]
        ...
        ...     def training_step(self, batch, batch_idx):
        ...         # The pre_processor will automatically apply the correct transform
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

        self.exportable_transform = get_exportable_transform(self.test_transform)

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Set the transforms for datamodule or dataloaders.

        The model-specific transforms are configured within PreProcessor and stored in
        model implementation. This method sets the transforms for the datamodule or
        dataloaders.

        Args:
            trainer: The Lightning trainer.
            pl_module: The Lightning module.
            stage: The stage (e.g., 'fit', 'train', 'val', 'test', 'predict').
        """
        super().setup(trainer, pl_module, stage)
        stage = TrainerFn(stage).value  # This is to convert the stage to a string
        stages = ["train", "val"] if stage == "fit" else [stage]
        for current_stage in stages:
            transform = getattr(self, f"{current_stage}_transform")
            if transform:
                if hasattr(trainer, "datamodule"):
                    set_datamodule_transform(trainer.datamodule, transform, current_stage)
                elif hasattr(trainer, f"{current_stage}_dataloaders"):
                    set_dataloader_transform(getattr(trainer, f"{current_stage}_dataloaders"), transform)
                else:
                    msg = f"Trainer does not have a datamodule or {current_stage}_dataloaders attribute"
                    raise ValueError(msg)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Apply transforms to the batch of tensors for inference.

        This forward-pass is only used after the model is exported.
        Within the Lightning training/validation/testing loops, the transforms are applied
        in the `on_*_batch_start` methods.
        """
        return self.exportable_transform(batch) if self.exportable_transform else batch

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
    ) -> None:
        """Apply transforms to the training batch."""
        del trainer, pl_module, batch_idx  # Unused parameters
        if self.train_transform:
            image, gt_mask = self.train_transform(batch.image, batch.gt_mask)
            batch.update(image=image, gt_mask=gt_mask)

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Apply transforms to the validation batch."""
        del trainer, pl_module, batch_idx, dataloader_idx  # Unused parameters
        if self.val_transform:
            image, gt_mask = self.val_transform(batch.image, batch.gt_mask)
            batch.update(image=image, gt_mask=gt_mask)

    def on_test_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Apply transforms to the test batch."""
        del trainer, pl_module, batch_idx, dataloader_idx  # Unused parameters
        if self.test_transform:
            image, gt_mask = self.test_transform(batch.image, batch.gt_mask)
            batch.update(image=image, gt_mask=gt_mask)

    def on_predict_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Apply transforms to the predict batch, which is the same as test batch."""
        self.on_test_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)
