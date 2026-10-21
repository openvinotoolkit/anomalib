# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Joint image and ROI preprocessing for GRD-Net."""

import torch
from lightning import Callback, LightningModule, Trainer
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import Resize, Transform

from anomalib.data import Batch
from anomalib.models.image.grdnet.data.dataclasses import GRDNetBatch
from anomalib.pre_processing import PreProcessor


class GRDNetPreProcessor(PreProcessor):
    """Transform GRD-Net images and masks while preserving their alignment.

    Args:
        transform: Joint torchvision v2 transform. ``None`` uses the canonical resize.

    Note:
        Checkpoints support the deterministic transforms registered by ``PreProcessor``.
        Unsupported random transforms are not restored; use dataset augmentations instead.
    """

    def __init__(self, transform: Transform | None = None) -> None:
        super().__init__(
            transform=transform
            if transform is not None
            else Resize(
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
        batch.image = batch.image.clamp(0.0, 1.0)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Resize and bound an image batch for exported inference.

        Args:
            batch: Input image batch.

        Returns:
            Resized image batch with values in ``[0, 1]``.
        """
        return super().forward(batch).clamp(0.0, 1.0)

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


class _GRDNetROIGuard(Callback):
    """Reject incompatible ROI preprocessing before a batch can be mutated."""

    @staticmethod
    def _check(pl_module: LightningModule, batch: Batch) -> None:
        """Require joint preprocessing for every ROI, including materialized fallbacks."""
        processor = pl_module.pre_processor
        if (
            getattr(batch, "roi_mask", None) is not None
            and processor is not None
            and not isinstance(processor, GRDNetPreProcessor)
        ):
            msg = (
                "GRD-Net ROI batches require GRDNetPreProcessor(transform=...) for joint image/mask transforms, "
                "a GRDNetPreProcessor subclass, or pre_processor=False with already aligned inputs."
            )
            raise ValueError(msg)

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
    ) -> None:
        """Check ROI compatibility before training preprocessing."""
        del trainer, batch_idx
        self._check(pl_module, batch)

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
    ) -> None:
        """Check ROI compatibility before validation preprocessing."""
        del trainer, batch_idx
        self._check(pl_module, batch)

    def on_test_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Check ROI compatibility before test preprocessing."""
        del trainer, batch_idx, dataloader_idx
        self._check(pl_module, batch)

    def on_predict_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Check ROI compatibility before prediction preprocessing."""
        del trainer, batch_idx, dataloader_idx
        self._check(pl_module, batch)
