# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Joint image and ROI preprocessing for GRD-Net."""

from lightning import LightningModule, Trainer
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import Resize

from anomalib.data import Batch
from anomalib.data.dataclasses.torch.grdnet import GRDNetBatch
from anomalib.pre_processing import PreProcessor


class GRDNetPreProcessor(PreProcessor):
    """Resize GRD-Net images and masks while preserving their alignment."""

    def __init__(self) -> None:
        super().__init__(
            transform=Resize(
                size=(256, 256),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
        )

    def _apply_transform(self, batch: Batch) -> None:
        """Apply the configured transform jointly to available batch fields."""
        if not self.transform:
            return
        if not isinstance(batch, GRDNetBatch) or batch.roi_mask is None:
            batch.image, batch.gt_mask = self.transform(batch.image, batch.gt_mask)
        else:
            batch.image, batch.gt_mask, batch.roi_mask = self.transform(batch.image, batch.gt_mask, batch.roi_mask)

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
    ) -> None:
        """Resize a training batch."""
        del trainer, pl_module, batch_idx
        self._apply_transform(batch)

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
    ) -> None:
        """Resize a validation batch."""
        del trainer, pl_module, batch_idx
        self._apply_transform(batch)

    def on_test_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Resize a test batch."""
        del trainer, pl_module, batch_idx, dataloader_idx
        self._apply_transform(batch)

    def on_predict_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Resize a prediction batch."""
        del trainer, pl_module, batch_idx, dataloader_idx
        self._apply_transform(batch)
