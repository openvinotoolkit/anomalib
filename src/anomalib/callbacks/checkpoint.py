"""Anomalib Model Checkpoint Callback."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint as LightningCheckpoint
from lightning.pytorch.trainer.states import TrainerFn

from anomalib import LearningType


class ModelCheckpoint(LightningCheckpoint):
    """Anomalib Model Checkpoint Callback.

    This class overrides the Lightning ModelCheckpoint callback to enable saving checkpoints without running any
    training steps. This is useful for zero-/few-shot models, where the fit sequence only consists of validation.

    To enable saving checkpoints without running any training steps, we need to override two checks which are being
    called in the ``on_validation_end`` method of the parent class:
    - ``_should_save_on_train_epoch_end``: This method checks whether the checkpoint should be saved at the end of a
        training epoch, or at the end of the validation sequence. We modify this method to default to saving at the end
        of the validation sequence when the model is of zero- or few-shot type, unless ``save_on_train_epoch_end`` is
        specifically set by the user.
    - ``_should_skip_saving_checkpoint``: This method checks whether the checkpoint should be saved at all. We modify
        this method to allow saving during both the ``FITTING`` and ``VALIDATING`` states. In addition, we allow saving
        if the global step has not changed since the last checkpoint, but only for zero- and few-shot models. This is
        needed because both the last global step and the last checkpoint remain unchanged during zero-/few-shot
        training, which would otherwise prevent saving checkpoints during validation.
    """

    def _should_skip_saving_checkpoint(self, trainer: Trainer) -> bool:
        """Checks whether the checkpoint should be saved.

        Overrides the parent method to allow saving during both the ``FITTING`` and ``VALIDATING`` states, and to allow
        saving when the global step and last_global_step_saved are both 0 (only for zero-/few-shot models).
        """
        is_zero_or_few_shot = trainer.lightning_module.learning_type in {LearningType.ZERO_SHOT, LearningType.FEW_SHOT}
        return (
            bool(trainer.fast_dev_run)  # disable checkpointing with fast_dev_run
            or trainer.state.fn not in {TrainerFn.FITTING, TrainerFn.VALIDATING}  # don't save anything during non-fit
            or trainer.sanity_checking  # don't save anything during sanity check
            or (self._last_global_step_saved == trainer.global_step and not is_zero_or_few_shot)
        )

    def _should_save_on_train_epoch_end(self, trainer: Trainer) -> bool:
        """Checks whether the checkpoint should be saved at the end of a training epoch or validation sequence.

        Overrides the parent method to default to saving at the end of the validation sequence when the model is of
        zero- or few-shot type, unless ``save_on_train_epoch_end`` is specifically set by the user.
        """
        if self._save_on_train_epoch_end is not None:
            return self._save_on_train_epoch_end

        if trainer.lightning_module.learning_type in {LearningType.ZERO_SHOT, LearningType.FEW_SHOT}:
            return False

        return super()._should_save_on_train_epoch_end(trainer)
