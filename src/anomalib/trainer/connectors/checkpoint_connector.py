"""Overrides the checkpoint connector to support saving/loading thresholding and normalization parameters."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.trainer.connectors import checkpoint_connector

from anomalib import trainer


class CheckpointConnector(checkpoint_connector.CheckpointConnector):
    """CheckpointConnector handles loading and saving checkpoints.

    Args:
        trainer (AnomalibTrainer): The trainer instance.
        resume_from_checkpoint (str, Path): Path to checkpoint to resume training from. This is deprecated in lightning
            and is present only for compatibility.
    """

    def __init__(self, trainer: "trainer.AnomalibTrainer", resume_from_checkpoint: _PATH | None = None) -> None:
        super().__init__(trainer, resume_from_checkpoint)
        self.trainer: "trainer.AnomalibTrainer"  # type: ignore

    def dump_checkpoint(self, weights_only: bool = False) -> dict:
        """Override dump checkpoint to save thresholding and normalization parameters.

        Args:
            weights_only (bool, optional): saving model weights only. Defaults to False.

        Returns:
            dict: Model checkpoint with image and pixel thresholding and normalization parameters.
        """
        checkpoint = super().dump_checkpoint(weights_only)
        checkpoint["image_threshold"] = self.trainer.image_threshold
        checkpoint["pixel_threshold"] = self.trainer.pixel_threshold
        if self.trainer.normalization_connector:
            checkpoint["normalization_metric"] = self.trainer.normalization_connector.metric
        return checkpoint

    def restore_training_state(self) -> None:
        """Loads the thresholding and normalization classes from the checkpoint."""
        super().restore_training_state()
        self._load_metrics()

    def _load_metrics(self) -> None:
        """Loads the thresholding and normalization classes from the checkpoint."""
        # get set of normalization keys in state dict
        checkpoint = self._loaded_checkpoint
        if "image_threshold" in checkpoint.keys():
            self.trainer.image_threshold = checkpoint["image_threshold"]
        if "pixel_threshold" in checkpoint.keys():
            self.trainer.pixel_threshold = checkpoint["pixel_threshold"]
        if "normalization_metric" in checkpoint.keys():
            assert (
                self.trainer.normalization_connector is not None
            ), "Normalizer not initialized while the checkpoint has a metric."
            self.trainer.normalization_connector.metric = checkpoint["normalization_metric"]
