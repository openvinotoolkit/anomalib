# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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

import logging
from typing import Any

import torch
from lightning import Callback, LightningModule, Trainer
from torch import nn
from torchvision.transforms.v2 import Transform

from anomalib.data import Batch

from .utils.spec import spec_to_transform, transform_to_spec
from .utils.transform import (
    get_exportable_transform,
)

logger = logging.getLogger(__name__)


class PreProcessor(nn.Module, Callback):
    """Anomalib pre-processor.

    This class serves as both a PyTorch module and a Lightning callback, handling
    the application of transforms to data batches as a pre-processing step.

    Args:
        transform (Transform | None): Transform to apply to the data before passing it to the model.

    Note:
        ``.transform`` may be any torchvision ``Transform``, but only a closed
        set of deterministic transforms (see
        :mod:`anomalib.pre_processing.utils._transform_registry`) can be
        persisted in a checkpoint. Randomized transforms such as
        ``RandomHorizontalFlip`` or ``RandomRotation`` belong in dataset
        *augmentations* (``train_augmentations`` etc.), not in a model's
        pre-processor; see the
        :doc:`Transforms guide </markdown/guides/how_to/data/transforms>` for
        why mixing the two is a common pitfall. An unsupported transform does
        not raise at construction; instead, ``checkpoint_config`` degrades
        gracefully (with a warning) so training is not interrupted, but the
        transform itself is not restored on reload.

    Example:
        >>> from torchvision.transforms.v2 import Compose, Resize, ToTensor
        >>> from anomalib.pre_processing import PreProcessor

        >>> # Define a custom set of transforms
        >>> transform = Compose([Resize((224, 224)), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        >>> # Pass the custom set of transforms to a model
        >>> pre_processor = PreProcessor(transform=transform)
        >>> model = MyModel(pre_processor=pre_processor)

        >>> # Advanced use: configure the default pre-processing behaviour of a Lightning module
        >>> class MyModel(LightningModule):
        ...     def __init__(self):
        ...         super().__init__()
        ...         ...
        ...
        ...     def configure_pre_processor(self):
        ...         transform = Compose([
        ...             Resize((224, 224)),
        ...             Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ...         ])
        ...         return PreProcessor(transform)
        ...
    """

    def __init__(
        self,
        transform: Transform | None = None,
    ) -> None:
        super().__init__()

        self.transform = transform
        self.export_transform = get_exportable_transform(self.transform)

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
    ) -> None:
        """Apply transforms to the batch of tensors during training."""
        del trainer, pl_module, batch_idx  # Unused
        if self.transform:
            batch.image, batch.gt_mask = self.transform(batch.image, batch.gt_mask)

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
    ) -> None:
        """Apply transforms to the batch of tensors during validation."""
        del trainer, pl_module, batch_idx  # Unused
        if self.transform:
            batch.image, batch.gt_mask = self.transform(batch.image, batch.gt_mask)

    def on_test_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Apply transforms to the batch of tensors during testing."""
        del trainer, pl_module, batch_idx, dataloader_idx  # Unused
        if self.transform:
            batch.image, batch.gt_mask = self.transform(batch.image, batch.gt_mask)

    def on_predict_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Apply transforms to the batch of tensors during prediction."""
        del trainer, pl_module, batch_idx, dataloader_idx  # Unused
        if self.transform:
            batch.image, batch.gt_mask = self.transform(batch.image, batch.gt_mask)

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

    def checkpoint_config(self) -> dict[str, Any]:
        """Get plain-data configuration to persist in a checkpoint.

        Uses ``getattr`` rather than ``self.transform`` directly so that a
        subclass which does not call ``super().__init__()`` (and therefore has
        no ``transform`` attribute) can still be checkpointed safely, instead of
        raising ``AttributeError`` during ``on_save_checkpoint``.

        ``.transform`` may hold a transform outside the safe-serialization
        registry (e.g. ``RandomHorizontalFlip``, which anomalib's own docs warn
        against using as a model-specific transform rather than a dataset
        augmentation). Rather than aborting the whole checkpoint save, this
        degrades gracefully: the transform is not persisted, a warning is
        logged, and the checkpoint still saves. The transform is also not
        restored on reload; see :meth:`load_checkpoint_config`.

        Override this method (together with :meth:`load_checkpoint_config`) in a
        subclass that manages additional or different transform state, for
        example separate per-stage transforms.

        Returns:
            dict[str, Any]: Plain-data configuration.
        """
        transform = getattr(self, "transform", None)
        try:
            spec = transform_to_spec(transform)
        except ValueError:
            logger.warning(
                "Cannot persist %s in a checkpoint under weights_only=True; it will not be restored on reload. "
                "Consider moving it to dataset augmentations (train_augmentations, etc.) instead of the "
                "model's pre-processor.",
                type(transform).__name__,
            )
            spec = None
        return {"transform": spec}

    def load_checkpoint_config(self, config: dict[str, Any]) -> None:
        """Restore configuration previously returned by :meth:`checkpoint_config`.

        Args:
            config (dict[str, Any]): Plain-data configuration to restore.
        """
        transform = spec_to_transform(config.get("transform"))
        self.transform = transform
        self.export_transform = get_exportable_transform(transform)
