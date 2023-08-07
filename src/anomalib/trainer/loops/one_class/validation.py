"""Validation loop for one-class classification."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from lightning.fabric.wrappers import _FabricDataLoader, _unwrap_objects
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning_utilities import apply_to_collection

from anomalib import trainer
from anomalib.trainer.loops.base import BaseLoop


class ValidationLoop(BaseLoop):
    def __init__(self, trainer: "trainer.AnomalibTrainer", verbose: bool = True):
        super().__init__(trainer, "validation")
        self.verbose = verbose

    def run_epoch_loop(self, val_dataloader) -> list[STEP_OUTPUT]:
        """Currently only runs one epoch."""
        self.trainer.fabric.call("on_validation_epoch_start", trainer=self, pl_module=self.model)

        outputs = self.run_batch_loop(val_dataloader)

        self.model.validation_epoch_end(outputs)
        self.trainer.fabric.call("validation_epoch_end", trainer=self, pl_module=self.model)

        return outputs

    def run_batch_loop(self, val_dataloader: _FabricDataLoader) -> list[STEP_OUTPUT]:
        outputs = []
        self.trainer.num_val_batches = [len(val_dataloader)]
        for batch_idx, batch in enumerate(val_dataloader):
            # end epoch if stopping training completely or max batches for this epoch reached
            if batch_idx >= self.trainer.limit_val_batches:
                break
            self.trainer.fabric.call(
                "on_validation_batch_start",
                batch=batch,
                batch_idx=batch_idx,
                dataloader_idx=0,
                trainer=self.trainer,
                pl_module=self.model,
            )

            output = self.model.validation_step(batch, batch_idx)
            output = apply_to_collection(output, torch.Tensor, lambda x: x.detach())

            output = self.model.validation_step_end(output)  # TODO change this

            self.trainer.fabric.call(
                "on_validation_batch_end",
                outputs=output,
                batch=batch,
                batch_idx=batch_idx,
                dataloader_idx=0,
                trainer=self.trainer,
                pl_module=self.model,
            )
            outputs.append(output)

        return outputs

    def setup(self):
        super().setup()
        torch.set_grad_enabled(False)
        self.model.eval()
        self.trainer.fabric.call("on_validation_start", trainer=self.trainer, pl_module=self.model)
        self.model.on_validation_start()

    def teardown(self):
        super().teardown()
        torch.set_grad_enabled(True)
        self.trainer.fabric.call("on_validation_end", trainer=self.trainer, pl_module=self.model)
        self.trainer.print_metrics(self.trainer.progress_bar_metrics, "val")
