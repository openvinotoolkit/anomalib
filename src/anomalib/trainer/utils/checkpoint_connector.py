"""Overrides the checkpoint connector to support saving/loading thresholding and normalization parameters."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.trainer.connectors import checkpoint_connector

import anomalib.trainer as trainer  # to avoid circular imports


class CheckpointConnector(checkpoint_connector.CheckpointConnector):
    def __init__(self, trainer: trainer.AnomalibTrainer, resume_from_checkpoint: _PATH | None = None) -> None:
        super().__init__(trainer, resume_from_checkpoint)
        self.trainer: trainer.AnomalibTrainer  # type: ignore

    def dump_checkpoint(self, weights_only: bool = False) -> dict:
        checkpoint = super().dump_checkpoint(weights_only)
        checkpoint["image_threshold"] = self.trainer.image_threshold
        checkpoint["pixel_threshold"] = self.trainer.pixel_threshold
        if self.trainer.normalizer:
            checkpoint["normalization_metric"] = self.trainer.normalizer.metric
        return checkpoint

    def restore_training_state(self) -> None:
        super().restore_training_state()
        self._load_metrics()

    def _load_metrics(self) -> None:
        """Loads the thersholding and normalization classes from the checkpoint."""
        # get set of normalization keys in state dict
        checkpoint = self._loaded_checkpoint
        if "image_threshold" in checkpoint.keys():
            self.trainer.image_threshold = checkpoint["image_threshold"]
        if "pixel_threshold" in checkpoint.keys():
            self.trainer.pixel_threshold = checkpoint["pixel_threshold"]
        if "normalization_metric" in checkpoint.keys():
            self.trainer.normalizer._metric = checkpoint["normalization_metric"]
