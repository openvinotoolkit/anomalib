"""Anomalib Model Checkpoint Callback.

This module provides the :class:`ModelCheckpoint` callback that extends PyTorch Lightning's
:class:`~lightning.pytorch.callbacks.ModelCheckpoint` to support zero-shot and few-shot learning scenarios.

The callback enables checkpoint saving without requiring training steps, which is particularly useful for
zero-shot and few-shot learning models where the training process may only involve validation.

Example:
    Create and use a checkpoint callback:

    >>> from anomalib.callbacks import ModelCheckpoint
    >>> checkpoint_callback = ModelCheckpoint(
    ...     dirpath="checkpoints",
    ...     filename="best",
    ...     monitor="val_loss"
    ... )
    >>> from lightning.pytorch import Trainer
    >>> trainer = Trainer(callbacks=[checkpoint_callback])

Note:
    This callback is particularly important for zero-shot and few-shot models where
    traditional training-based checkpoint saving strategies may not be appropriate.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint as LightningCheckpoint
from lightning.pytorch.trainer.states import TrainerFn

from anomalib import LearningType


class ModelCheckpoint(LightningCheckpoint):
    """Custom ModelCheckpoint callback for Anomalib.

    This callback extends PyTorch Lightning's
    :class:`~lightning.pytorch.callbacks.ModelCheckpoint` to enable checkpoint saving
    without requiring training steps. This is particularly useful for zero-shot and few-shot
    learning models where the training process may only involve validation.

    The callback overrides two key methods from the parent class:

    1. :meth:`_should_save_on_train_epoch_end`: Controls whether checkpoints are saved at the end
       of training epochs or validation sequences. For zero-shot and few-shot models, it defaults
       to saving at validation end unless explicitly configured otherwise.

    2. :meth:`_should_skip_saving_checkpoint`: Determines if checkpoint saving should be skipped.
       Modified to:

       - Allow saving during both ``FITTING`` and ``VALIDATING`` states
       - Permit saving even when global step hasn't changed (for zero-shot/few-shot models)
       - Maintain standard checkpoint skipping conditions (``fast_dev_run``, sanity checking)

    Example:
        Create and use a checkpoint callback:

        >>> from anomalib.callbacks import ModelCheckpoint
        >>> # Create a checkpoint callback
        >>> checkpoint_callback = ModelCheckpoint(
        ...     dirpath="checkpoints",
        ...     filename="best",
        ...     monitor="val_loss"
        ... )
        >>> # Use it with Lightning Trainer
        >>> from lightning.pytorch import Trainer
        >>> trainer = Trainer(callbacks=[checkpoint_callback])

    Note:
        All arguments from PyTorch Lightning's :class:`~lightning.pytorch.callbacks.ModelCheckpoint` are supported.
        See :class:`~lightning.pytorch.callbacks.ModelCheckpoint` for details.
    """

    def _should_skip_saving_checkpoint(self, trainer: Trainer) -> bool:
        """Determine if checkpoint saving should be skipped.

        Args:
            trainer (:class:`~lightning.pytorch.Trainer`): PyTorch Lightning trainer instance.

        Returns:
            bool: ``True`` if checkpoint saving should be skipped, ``False`` otherwise.

        Note:
            The method considers the following conditions:

            - Skips if ``fast_dev_run`` is enabled
            - Skips if not in ``FITTING`` or ``VALIDATING`` state
            - Skips during sanity checking
            - For non-zero/few-shot models, skips if global step hasn't changed
        """
        is_zero_or_few_shot = trainer.lightning_module.learning_type in {LearningType.ZERO_SHOT, LearningType.FEW_SHOT}
        return (
            bool(trainer.fast_dev_run)  # disable checkpointing with fast_dev_run
            or trainer.state.fn not in {TrainerFn.FITTING, TrainerFn.VALIDATING}  # don't save anything during non-fit
            or trainer.sanity_checking  # don't save anything during sanity check
            or (self._last_global_step_saved == trainer.global_step and not is_zero_or_few_shot)
        )

    def _should_save_on_train_epoch_end(self, trainer: Trainer) -> bool:
        """Determine if checkpoint should be saved at training epoch end.

        Args:
            trainer (:class:`~lightning.pytorch.Trainer`): PyTorch Lightning trainer instance.

        Returns:
            bool: ``True`` if checkpoint should be saved at training epoch end, ``False`` otherwise.

        Note:
            The method follows this decision flow:

            - Returns user-specified value if ``_save_on_train_epoch_end`` is set
            - For zero/few-shot models, defaults to ``False`` (save at validation end)
            - Otherwise, follows parent class behavior
        """
        if self._save_on_train_epoch_end is not None:
            return self._save_on_train_epoch_end

        if trainer.lightning_module.learning_type in {LearningType.ZERO_SHOT, LearningType.FEW_SHOT}:
            return False

        return super()._should_save_on_train_epoch_end(trainer)
