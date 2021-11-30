"""Base Anomaly Module for Training Task."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import List, Union

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.callbacks.base import Callback
from torch import nn
from torchmetrics import F1, MetricCollection

from anomalib.core.metrics import AUROC, OptimalF1


class AnomalyModule(pl.LightningModule):
    """AnomalyModule to train, validate, predict and test images.

    Args:
        params (Union[DictConfig, ListConfig]): Configuration
    """

    def __init__(self, params: Union[DictConfig, ListConfig]):

        super().__init__()
        # Force the type for hparams so that it works with OmegaConfig style of accessing
        self.hparams: Union[DictConfig, ListConfig]  # type: ignore
        self.save_hyperparameters(params)
        self.loss: torch.Tensor
        self.callbacks: List[Callback]
        self.register_buffer("threshold", torch.tensor(params.model.threshold.default))  # pylint: disable=not-callable
        self.threshold: torch.Tensor

        self.model: nn.Module

        # metrics
        self.image_metrics = MetricCollection(
            [AUROC(num_classes=1, pos_label=1, compute_on_step=False)], prefix="image_"
        )
        if params.model.threshold.adaptive:
            self.image_metrics.add_metrics([OptimalF1(num_classes=1)])
        else:
            self.image_metrics.add_metrics([F1(num_classes=1, compute_on_step=False, threshold=self.threshold.item())])
        if self.hparams.dataset.task == "segmentation":
            self.pixel_metrics = self.image_metrics.clone(prefix="pixel_")

    def forward(self, batch):  # pylint: disable=arguments-differ
        """Forward-pass input tensor to the module.

        Args:
            batch (Tensor): Input Tensor

        Returns:
            [Tensor]: Output tensor from the model.
        """
        return self.model(batch)

    def predict_step(self, batch, batch_idx, _):  # pylint: disable=arguments-differ, signature-differs
        """Step function called during :meth:`~pytorch_lightning.trainer.trainer.Trainer.predict`.

        By default, it calls :meth:`~pytorch_lightning.core.lightning.LightningModule.forward`.
        Override to add any processing logic.

        Args:
            batch: Current batch
            batch_idx: Index of current batch
            dataloader_idx: Index of the current dataloader

        Return:
            Predicted output
        """
        return self._post_process(self.validation_step(batch, batch_idx), predict_labels=True)

    def test_step(self, batch, _):  # pylint: disable=arguments-differ
        """Calls validation_step for anomaly map/score calculation.

        Args:
          batch: Input batch
          _: Index of the batch.

        Returns:
          Dictionary containing images, features, true labels and masks.
          These are required in `validation_epoch_end` for feature concatenation.
        """
        return self.validation_step(batch, _)

    def validation_step_end(self, val_step_outputs):  # pylint: disable=arguments-differ
        """Called at the end of each validation step."""
        val_step_outputs = self._post_process(val_step_outputs)
        self.image_metrics(val_step_outputs["pred_scores"], val_step_outputs["label"].int())
        if self.hparams.dataset.task == "segmentation":
            self.pixel_metrics(val_step_outputs["anomaly_maps"].flatten(), val_step_outputs["mask"].flatten().int())
        return val_step_outputs

    def test_step_end(self, test_step_outputs):  # pylint: disable=arguments-differ
        """Called at the end of each test step."""
        return self.validation_step_end(test_step_outputs)

    def validation_epoch_end(self, _outputs):
        """Compute threshold and performance metrics.

        Args:
          outputs: Batch of outputs from the validation step
        """
        if self.hparams.model.threshold.adaptive:
            self.image_metrics.compute()
            self.threshold = self.image_metrics.OptimalF1.threshold
        self._log_metrics()

    def test_epoch_end(self, _outputs):
        """Compute and save anomaly scores of the test set.

        Args:
            outputs: Batch of outputs from the validation step
        """
        self._log_metrics()

    def _post_process(self, outputs, predict_labels=False):
        """Compute labels based on model predictions."""
        if "pred_scores" not in outputs and "anomaly_maps" in outputs:
            outputs["pred_scores"] = (
                outputs["anomaly_maps"].reshape(outputs["anomaly_maps"].shape[0], -1).max(axis=1).values
            )
        if predict_labels:
            outputs["pred_labels"] = outputs["pred_scores"] >= self.threshold.item()
        return outputs

    def _log_metrics(self):
        """Log computed performance metrics."""
        self.log_dict(self.image_metrics)
        if self.hparams.dataset.task == "segmentation":
            self.log_dict(self.pixel_metrics)
