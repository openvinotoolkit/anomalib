"""Anomalib pre-processing module."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from lightning import Callback, LightningModule, Trainer
from torch import nn
from torchvision.transforms.v2 import Transform

from anomalib.data.dataclasses.torch.base import Batch


class PreProcessor(nn.Module, Callback):
    """Anomalib pre-processor."""

    def __init__(
        self,
        train_transforms: Transform | None = None,
        val_transforms: Transform | None = None,
        test_transforms: Transform | None = None,
        transforms: Transform | None = None,
    ) -> None:
        super().__init__()

        if transforms and any([train_transforms, val_transforms, test_transforms]):
            msg = (
                "`transforms` cannot be used together with `train_transforms`, `val_transforms`, `test_transforms`.\n"
                "If you want to apply the same transform to the training, validation and test data, "
                "use only `transforms`. \n"
                "Otherwise, specify transforms for training, validation and test individually."
            )
            raise ValueError(msg)

        self.train_transforms = train_transforms or transforms
        self.val_transforms = val_transforms or transforms
        self.test_transforms = test_transforms or transforms

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
    ) -> None:
        """Apply transforms to the training batch."""
        del trainer, pl_module, batch_idx  # Unused parameters

        if self.train_transforms:
            image, gt_mask = self.train_transforms(batch.image, batch.gt_mask)
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

        if self.val_transforms:
            image, gt_mask = self.val_transforms(batch.image, batch.gt_mask)
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

        if self.test_transforms:
            image, gt_mask = self.test_transforms(batch.image, batch.gt_mask)
            batch.update(image=image, gt_mask=gt_mask)

    def on_predict_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Apply transforms to the predict batch."""
        del trainer, pl_module, batch_idx, dataloader_idx  # Unused parameters

        if self.test_transforms:
            image, gt_mask = self.test_transforms(batch.image, batch.gt_mask)
            batch.update(image=image, gt_mask=gt_mask)
