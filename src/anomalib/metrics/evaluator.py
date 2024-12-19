"""Evaluator module for LightningModule.

The Evaluator module computes and logs metrics during validation and test steps.
Each ``AnomalibModule`` should have an Evaluator module as a submodule to compute
and log metrics. An Evaluator module can be passed to the ``AnomalibModule`` as a
parameter during initialization. When no Evaluator module is provided, the
``AnomalibModule`` will use a default Evaluator module that logs a default set of
metrics.

Args:
    val_metrics (Sequence[AnomalibMetric] | AnomalibMetric | None, optional):
        Validation metrics. Defaults to ``None``.
    test_metrics (Sequence[AnomalibMetric] | AnomalibMetric | None, optional):
        Test metrics. Defaults to ``None``.
    compute_on_cpu (bool, optional): Whether to compute metrics on CPU.
        Defaults to ``True``.

Example:
    >>> from anomalib.metrics import F1Score, AUROC
    >>> from anomalib.data import ImageBatch
    >>> import torch
    >>>
    >>> # Initialize metrics with fields to use from batch
    >>> f1_score = F1Score(fields=["pred_label", "gt_label"])
    >>> auroc = AUROC(fields=["pred_score", "gt_label"])
    >>>
    >>> # Create evaluator with test metrics
    >>> evaluator = Evaluator(test_metrics=[f1_score, auroc])
    >>>
    >>> # Create sample batch
    >>> batch = ImageBatch(
    ...     image=torch.rand(4, 3, 256, 256),
    ...     pred_label=torch.tensor([0, 0, 1, 1]),
    ...     gt_label=torch.tensor([0, 0, 1, 1]),
    ...     pred_score=torch.tensor([0.1, 0.2, 0.8, 0.9])
    ... )
    >>>
    >>> # Update metrics with batch
    >>> evaluator.on_test_batch_end(None, None, None, batch, 0)
    >>>
    >>> # Compute and log metrics at end of epoch
    >>> evaluator.on_test_epoch_end(None, None)

Note:
    The evaluator will automatically move metrics to CPU for computation if
    ``compute_on_cpu=True`` and only one device is used. For multi-GPU training,
    ``compute_on_cpu`` is automatically set to ``False``.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence
from typing import Any

from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import ModuleList
from torchmetrics import Metric

from anomalib.metrics import AnomalibMetric

logger = logging.getLogger(__name__)


class Evaluator(nn.Module, Callback):
    """Evaluator module for LightningModule.

    The Evaluator module is a PyTorch module that computes and logs metrics during
    validation and test steps. Each AnomalibModule should have an Evaluator module as
    a submodule to compute and log metrics during validation and test steps. An Evaluation
    module can be passed to the AnomalibModule as a parameter during initialization. When
    no Evaluator module is provided, the AnomalibModule will use a default Evaluator module
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
        val_metrics: AnomalibMetric | Sequence[AnomalibMetric] | None = None,
        test_metrics: AnomalibMetric | Sequence[AnomalibMetric] | None = None,
        compute_on_cpu: bool = True,
    ) -> None:
        super().__init__()
        self.val_metrics = ModuleList(self.validate_metrics(val_metrics))
        self.test_metrics = ModuleList(self.validate_metrics(test_metrics))
        self.compute_on_cpu = compute_on_cpu

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Move metrics to cpu if ``num_devices == 1`` and ``compute_on_cpu`` is set to ``True``."""
        del pl_module, stage  # Unused arguments.
        if trainer.num_devices > 1:
            if self.compute_on_cpu:
                logger.warning("Number of devices is greater than 1, setting compute_on_cpu to False.")
        elif self.compute_on_cpu:
            self.metrics_to_cpu(self.val_metrics)
            self.metrics_to_cpu(self.test_metrics)

    @staticmethod
    def validate_metrics(metrics: AnomalibMetric | Sequence[AnomalibMetric] | None) -> Sequence[AnomalibMetric]:
        """Validate metrics."""
        if metrics is None:
            return []
        if isinstance(metrics, AnomalibMetric):
            return [metrics]
        if not isinstance(metrics, Sequence):
            msg = f"metrics must be an AnomalibMetric or a list of AnomalibMetrics, got {type(metrics)}"
            raise TypeError(msg)
        return metrics

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
