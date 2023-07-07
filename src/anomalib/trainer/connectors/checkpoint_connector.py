"""Saving and Loading the model + trainer params from checkpoint"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

from lightning_utilities.core.rank_zero import rank_zero_info

from anomalib import trainer


class CheckpointConnector:
    def __init__(self, trainer: "trainer.AnomalibTrainer", checkpoint_dir: Path) -> None:
        self.trainer = trainer
        self.checkpoint_dir = checkpoint_dir

    # def dump_checkpoint(self) -> dict:
    #     pass

    def restore(self, state: dict | None, filepath: Path | None = None) -> None:
        if filepath is None:
            filepath = self._select_ckpt_path()

        if filepath is None or not filepath.exists():
            rank_zero_info(f"No checkpoint found at {filepath}. Skipping restore.")
        else:
            if state is None:
                state = dict()
            remainder = self.trainer.fabric.load(filepath, state)
            self.trainer.global_step = remainder.pop("global_step")
            self.trainer.current_epoch = remainder.pop("current_epoch")
            rank_zero_info(f"Restored checkpoint from {filepath}.")

            if remainder:
                raise RuntimeError(f"Unused Checkpoint Values: {remainder}")

    def _select_ckpt_path(self) -> Path | None:
        # Intended to be similar to lightning's checkpoint connector
        file_path = None
        if (self.checkpoint_dir / "best.ckpt").exists():
            file_path = self.checkpoint_dir / "best.ckpt"
        else:
            # get the latest checkpoint of format epoch-num.ckpt
            file_path = max(self.checkpoint_dir.glob("*.ckpt"), key=lambda x: int(x.stem.split("-")[1]), default=None)
        return file_path

    def save(self, state: dict | None) -> None:
        """Saves a checkpoint to the ``checkpoint_dir``

        Args:
            state: A mapping containing model, optimizer and lr scheduler.
        """
        if state is None:
            state = dict()

        state.update(global_step=self.trainer.global_step, current_epoch=self.trainer.current_epoch)

        self.trainer.fabric.save(self.checkpoint_dir / f"epoch-{self.trainer.current_epoch:04d}.ckpt", state)
