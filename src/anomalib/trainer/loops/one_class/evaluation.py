"""Base class for evaluation loops."""


import logging
from typing import Any

import torch
from lightning.fabric.wrappers import _FabricDataLoader, _unwrap_objects
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning_utilities import apply_to_collection
from torch.utils.data import DataLoader

from anomalib import trainer
from anomalib.trainer.loops.base import BaseLoop

logger = logging.getLogger(__name__)


class EvaluationLoop(BaseLoop):
    def __init__(self, trainer: "trainer.AnomalibTrainer", stage: TrainerFn, verbose: bool = True):
        super().__init__(trainer, stage)
        self.verbose = verbose
        self.dataloader: _FabricDataLoader

    def run_epoch_loop(self) -> list[STEP_OUTPUT]:
        """Currently only runs one epoch."""

        if self.stage != TrainerFn.PREDICTING:
            self.trainer.fabric.call(f"on_{self.stage}_epoch_start", trainer=self, pl_module=self.model)

        outputs = self.run_batch_loop(self.dataloader)

        if self.stage != TrainerFn.PREDICTING:
            self._call_impl(self.model, f"{self.stage}_epoch_end", outputs=outputs)
            self.trainer.fabric.call(f"{self.stage}_epoch_end", trainer=self, pl_module=self.model)

        return outputs

    def run_batch_loop(self, dataloader: _FabricDataLoader) -> list[STEP_OUTPUT]:
        outputs = []
        setattr(
            self.trainer, f"num_{self.stage}_batches", [len(dataloader)]
        )  # num_val_batches does not work as it is now num_validation_batches
        for batch_idx, batch in enumerate(dataloader):
            # end epoch if stopping training completely or max batches for this epoch reached
            if batch_idx >= getattr(self.trainer, f"limit_{self.stage}_batches"):
                break
            self.trainer.fabric.call(
                f"on_{self.stage}_batch_start",
                batch=batch,
                batch_idx=batch_idx,
                dataloader_idx=0,
                trainer=self.trainer,
                pl_module=self.model,
            )

            output = self._call_impl(self.model, f"{self.stage}_step", batch, batch_idx)
            output = apply_to_collection(output, torch.Tensor, lambda x: x.detach())

            if self.stage != TrainerFn.PREDICTING:
                output = self._call_impl(self.model, f"{self.stage}_step_end", output)

            self.trainer.fabric.call(
                f"on_{self.stage}_batch_end",
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
        """Setup the evaluation loop."""
        super().setup()
        self._connect_dataloader()
        torch.set_grad_enabled(False)
        self.model.eval()
        self.trainer.fabric.call(f"on_{self.stage}_start", trainer=self.trainer, pl_module=self.model)
        self._call_impl(self.model, f"on_{self.stage}_start")

    def teardown(self):
        super().teardown()
        torch.set_grad_enabled(True)
        self.trainer.fabric.call(f"on_{self.stage}_end", trainer=self.trainer, pl_module=self.model)
        if self.verbose:
            self.trainer.print_metrics(self.trainer.progress_bar_metrics, self.stage)

    def _call_impl(self, module: Any, method: str, *args, **kwargs):
        """Call a method on a module."""
        _method = getattr(module, method, None)
        value = None
        if method is None:
            logger.error(f"Method {method} not found on {module}. Skipping.")
        else:
            value = _method(*args, **kwargs)
        return value

    def _connect_dataloader(self):
        self.dataloader = getattr(self.trainer, f"{self.stage}_dataloaders")[0]
