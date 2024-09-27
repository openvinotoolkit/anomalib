"""Evaluator module for LightningModule."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any

from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import ModuleList
from torchmetrics import Metric

from anomalib.metrics import AnomalibMetric


class Evaluator(nn.Module, Callback):
    """Evaluator module for LightningModule.

    The Evaluator module is a PyTorch module that computes and logs metrics during
    validation and test steps. Each AnomalyModule should have an Evaluator module as
    a submodule to compute and log metrics during validation and test steps. An Evaluation
    module can be passed to the AnomalyModule as a parameter during initialization. When
    no Evaluator module is provided, the AnomalyModule will use a default Evaluator module
    that logs a default set of metrics.

    Args:
        val_metrics (Sequence[AnomalibMetric], optional): Validation metrics.
            Defaults to ``[]``.
        test_metrics (Sequence[AnomalibMetric], optional): Test metrics.
            Defaults to ``[]``.
        compute_on_cpu (bool, optional): Whether to compute metrics on CPU.
            Defaults to ``True``.

    Examples:
        >>> from anomalib.metrics import F1Score, AUROC
        >>> from anomalib.data import ImageBatch
        >>> import torch
        >>>
        >>> f1_score = F1Score(fields=["pred_label", "gt_label"])
        >>> auroc = AUROC(fields=["pred_score", "gt_label"])
        >>>
        >>> evaluator = Evaluator(test_metrics=[f1_score])
    """

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
        """Update validation metrics with the batch output."""
        del trainer, outputs, batch_idx, dataloader_idx, pl_module  # Unused arguments.
        for metric in self.val_metrics:
            metric.update(batch)

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Compute and log validation metrics."""
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
        """Update test metrics with the batch output."""
        del trainer, outputs, batch_idx, dataloader_idx, pl_module  # Unused arguments.
        for metric in self.test_metrics:
            metric.update(batch)

    def on_test_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Compute and log test metrics."""
        del trainer, pl_module  # Unused argument.
        for metric in self.test_metrics:
            self.log(metric.name, metric)

    def metrics_to_cpu(self, metrics: Metric | list[Metric] | ModuleList) -> None:
        """Set the compute_on_cpu attribute of the metrics to True."""
        if isinstance(metrics, Metric):
            metrics.compute_on_cpu = True
        elif isinstance(metrics, (list | ModuleList)):
            for metric in metrics:
                self.metrics_to_cpu(metric)
        else:
            msg = f"metrics must be a Metric or a list of metrics, got {type(metrics)}"
            raise TypeError(msg)
