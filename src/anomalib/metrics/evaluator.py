from collections.abc import Sequence
from typing import Any

from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import ModuleList
from torchmetrics import Metric

from anomalib.metrics import AnomalibMetric


class Evaluator(nn.Module, Callback):
    def __init__(
        self,
        val_metrics: Sequence[AnomalibMetric] = [],
        test_metrics: Sequence[AnomalibMetric] = [],
        compute_on_cpu: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.val_metrics = ModuleList(val_metrics)
        self.test_metrics = ModuleList(test_metrics)
        if compute_on_cpu:
            self.metrics_to_cpu(self.val_metrics)
            self.metrics_to_cpu(self.test_metrics)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, outputs, batch_idx, dataloader_idx, pl_module  # Unused arguments.
        for metric in self.val_metrics:
            metric.update(batch)

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        del trainer, pl_module  # Unused argument.
        for metric in self.val_metrics:
            self.log(metric.name, metric)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, outputs, batch_idx, dataloader_idx, pl_module  # Unused arguments.
        for metric in self.test_metrics:
            metric.update(batch)

    def on_test_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        del trainer, pl_module  # Unused argument.
        for metric in self.test_metrics:
            self.log(metric.name, metric)

    def metrics_to_cpu(self, metrics: Metric | list[Metric] | ModuleList) -> None:
        if isinstance(metrics, Metric):
            metrics.compute_on_cpu = True
        elif isinstance(metrics, (list, ModuleList)):
            for metric in metrics:
                self.metrics_to_cpu(metric)
        else:
            msg = f"metrics must be a Metric or a list of metrics, got {type(metrics)}"
            raise TypeError(msg)
