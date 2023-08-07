"""Fit loop for one-class anomaly detection."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from typing import Any

import torch
from lightning.fabric.wrappers import _FabricDataLoader, _unwrap_objects
from lightning.pytorch.trainer.states import TrainerStatus
from lightning_utilities import apply_to_collection

from anomalib import trainer
from anomalib.models import AnomalyModule
from anomalib.trainer.loops.base import BaseLoop


class FitLoop(BaseLoop):
    def __init__(self, trainer: "trainer.AnomalibTrainer", min_epochs: int | None = 0, max_epochs: int | None = None):
        super().__init__(trainer, "fit")
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.val_loop: BaseLoop

    @property
    def current_epoch(self):
        return self.trainer.current_epoch

    @current_epoch.setter
    def current_epoch(self, value):
        self.trainer.current_epoch = value

    def run_epoch_loop(
        self,
        train_dataloader,
        val_dataloader,
    ):
        # check if we even need to train here
        if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
            self.trainer.should_stop = True

        while not self.trainer.should_stop:
            self.trainer.num_training_batches = min(len(train_dataloader), self.trainer.limit_train_batches)
            self.trainer.fabric.call("on_train_epoch_start", trainer=self, pl_module=self.model)
            self.run_batch_loop(train_dataloader)

            self.trainer.step_scheduler(
                self.trainer.model, self.trainer.scheduler_cfg, level="epoch", current_value=self.current_epoch
            )

            self.trainer.checkpoint_connector.save()
            self.trainer.fabric.call("on_train_epoch_end", trainer=self.trainer, pl_module=self.model)

            if len(self.trainer._current_train_return) > 0:
                self.trainer._current_train_return = self.model.training_epoch_end(self.trainer._current_train_return)
            # run validation at end of training
            if self.trainer.should_validate:
                # self.trainer.validating = True
                with torch.no_grad():
                    self.trainer.validating = True
                    self.val_loop.run(val_dataloader)
                    self.trainer.training = True

            self.current_epoch += 1
            # stopping condition on epoch level
            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.trainer.should_stop = True

        self.trainer.state.status = TrainerStatus.FINISHED

    def run_batch_loop(self, train_dataloader: _FabricDataLoader):
        iterable = train_dataloader
        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.trainer.should_stop or batch_idx >= self.trainer.limit_train_batches:
                return

            self.trainer.fabric.call(
                "on_train_batch_start",
                batch=batch,
                batch_idx=batch_idx,
                trainer=self.trainer,
                pl_module=self.model,
            )

            # check if optimizer should step in gradient accumulation
            should_optim_step = (
                self.trainer.global_step % self.trainer.grad_accum_steps == 0 and self.trainer.optimizer is not None
            )
            if should_optim_step:
                # currently only supports a single optimizer
                self.trainer.fabric.call("on_before_optimizer_step", self.trainer.optimizer, 0)

                # optimizer step runs train step internally through closure
                self.trainer.optimizer.step(
                    partial(self.step, model=self.trainer.model, batch=batch, batch_idx=batch_idx)
                )
                self.trainer.fabric.call("on_before_zero_grad", self.trainer.optimizer)

                self.trainer.optimizer.zero_grad()

            else:
                # gradient accumulation -> no optimizer step
                self.step(model=self.trainer.model, batch=batch, batch_idx=batch_idx)

            self.trainer.fabric.call(
                "on_train_batch_end",
                outputs=self.trainer._current_train_return,
                batch=batch,
                batch_idx=batch_idx,
                trainer=self.trainer,
                pl_module=self.model,
            )

            # this guard ensures, we only step the scheduler once per global step
            if should_optim_step:
                self.trainer.step_scheduler(
                    self.trainer.model, self.trainer.scheduler_cfg, level="step", current_value=self.trainer.global_step
                )

            # only increase global step if optimizer stepped
            self.trainer.global_step += int(should_optim_step)

            # stopping criterion on step level
            if self.trainer.max_steps is not None and self.trainer.global_step >= self.trainer.max_steps:
                self.should_stop = True
                break

    def step(self, model: AnomalyModule, batch: Any, batch_idx: int) -> torch.Tensor | None:
        """A single training step, running forward and backward. The optimizer step is called separately, as this
        is given as a closure to the optimizer step.

        Args:
            model: the lightning module to train
            batch: the batch to run the forward on
            batch_idx: index of the current batch w.r.t the current epoch
        """
        outputs: torch.Tensor | dict[str, Any] = model.training_step(batch, batch_idx=batch_idx)
        if outputs is None:
            return None

        loss = outputs if isinstance(outputs, torch.Tensor) else outputs["loss"]

        self.trainer.fabric.call("on_before_backward", loss)
        self.trainer.fabric.backward(loss)
        self.trainer.fabric.call("on_after_backward")

        # avoid gradients in stored/accumulated values -> prevents potential OOM
        self.trainer._current_train_return = apply_to_collection(
            outputs, dtype=torch.Tensor, function=lambda x: x.detach()
        )

        return loss

    def setup(self):
        """Connects the validation loop to the fit loop."""
        super().setup()
        torch.set_grad_enabled(True)
        self.trainer.model.train()
        self.val_loop = self.trainer.validation_loop
        self.trainer.fabric.call("on_train_start", trainer=self.trainer, pl_module=self.model)

    def teardown(self):
        super().teardown()
        self.trainer.fabric.call("on_train_end", trainer=self.trainer, pl_module=self.model)
