"""Saving and Loading the model + trainer params from checkpoint"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging
from pathlib import Path

from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_warn

import anomalib
from anomalib import trainer

log = logging.getLogger(__name__)


class CheckpointConnector:
    """Handles saving and loading checkpoints.

    Args:
        trainer: The trainer instance.
        ckpt_path: If path to the weights is given, it loads the weights from the path.
    """

    def __init__(self, trainer: "trainer.AnomalibTrainer", ckpt_path: Path) -> None:
        self.trainer = trainer
        self.ckpt_path = ckpt_path
        self._initialize()

    def _initialize(self) -> None:
        """Creates checkpoint directory if it doesn't exist."""
        if not self.ckpt_path.exists():
            self.ckpt_path.mkdir(parents=True, exist_ok=True)

    def dump_checkpoint(self) -> dict:
        """Create a checkpoint dictionary of the current state.

        Return:
            A dictionary containing the current state of the trainer.
        """
        assert self.trainer.model is not None, "Model is not initialized."
        checkpoint = {
            "anomalib_version": anomalib.__version__,
            "model": self.trainer.model.state_dict(),
            "loggers": self.trainer.loggers,
            "callbacks": self.trainer.callbacks,
            "global_step": self.trainer.global_step,
            "current_epoch": self.trainer.current_epoch,
            "grad_accum_steps": self.trainer.grad_accum_steps,
            "seed": self.trainer.seed,
            "max_epochs": self.trainer.max_epochs,
            "max_steps": self.trainer.max_steps,
            "limit_train_batches": self.trainer.limit_train_batches,
            "limit_val_batches": self.trainer.limit_validation_batches,
            "limit_test_batches": self.trainer.limit_test_batches,
        }
        # TODO add loops
        return checkpoint

    def restore(self) -> None:
        filepath = self._select_ckpt_path()
        if filepath is None:
            if self.trainer.training:
                rank_zero_info("Training called without existing checkpoint. Skipping restore.")
            else:
                # TODO get trainer stage and use it's value in the message
                raise RuntimeError("No checkpoint found. Checkpoint is necessary when not training.")
        else:
            rank_zero_info("Restoring checkpoint from %s", filepath)

            state = self.dump_checkpoint()
            remainder = self.trainer.fabric.load(filepath, state)
            self.trainer.global_step = state.pop("global_step")
            self.trainer.current_epoch = state.pop("current_epoch")
            # TODO refactor to save and load trainer state
            checkpoint_anomalib_version = state.pop("anomalib_version")
            if checkpoint_anomalib_version != anomalib.__version__:
                rank_zero_warn(
                    f"Current Anomalib version {anomalib.__version__} is different from the checkpoint version"
                    f" {checkpoint_anomalib_version}."
                )
            if remainder:
                raise RuntimeError(f"Unused Checkpoint Values: {remainder}")

    def _select_ckpt_path(self) -> Path | None:
        # Intended to be similar to lightning's checkpoint connector
        file_path = None
        if self.ckpt_path.is_file():
            file_path = self.ckpt_path
        elif self.ckpt_path.is_dir():
            if (self.ckpt_path / "best.ckpt").exists():
                file_path = self.ckpt_path / "best.ckpt"
            else:
                # get the latest checkpoint of format epoch-num.ckpt
                file_path = max(self.ckpt_path.glob("*.ckpt"), key=lambda x: int(x.stem.split("-")[1]), default=None)
        return file_path

    def save(self) -> None:
        """Saves a checkpoint to the ``checkpoint_dir``"""

        ckpt_path = self.ckpt_path
        if ckpt_path.is_file():
            ckpt_path = ckpt_path.parent
        state = self.dump_checkpoint()
        self.trainer.fabric.save(ckpt_path / f"epoch-{self.trainer.current_epoch:04d}.ckpt", state)
