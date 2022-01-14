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
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.callbacks.base import Callback
from torch import Tensor, nn
from torchmetrics import F1, MetricCollection

from anomalib.core.metrics import (
    AUROC,
    AdaptiveThreshold,
    AnomalyScoreDistribution,
    MinMax,
)


class AnomalyModule(pl.LightningModule):
    """AnomalyModule to train, validate, predict and test images.

    Acts as a base class for all the Anomaly Modules in the library.

    Args:
        params (Union[DictConfig, ListConfig]): Configuration
    """

    def __init__(self, params: Union[DictConfig, ListConfig]):

        super().__init__()
        # Force the type for hparams so that it works with OmegaConfig style of accessing
        self.hparams: Union[DictConfig, ListConfig]  # type: ignore
        self.save_hyperparameters(params)
        self.loss: Tensor
        self.callbacks: List[Callback]

        self.image_threshold = AdaptiveThreshold(self.hparams.model.threshold.image_default).cpu()
        self.pixel_threshold = AdaptiveThreshold(self.hparams.model.threshold.pixel_default).cpu()

        self.training_distribution = AnomalyScoreDistribution().cpu()
        self.min_max = MinMax().cpu()

        self.model: nn.Module

        # metrics
        auroc = AUROC(num_classes=1, pos_label=1, compute_on_step=False)
        f1_score = F1(num_classes=1, compute_on_step=False)
        self.image_metrics = MetricCollection([auroc, f1_score], prefix="image_").cpu()
        self.pixel_metrics = self.image_metrics.clone(prefix="pixel_").cpu()

    def forward(self, batch):  # pylint: disable=arguments-differ
        """Forward-pass input tensor to the module.

        Args:
            batch (Tensor): Input Tensor

        Returns:
            Tensor: Output tensor from the model.
        """
        return self.model(batch)

    def validation_step(self, batch, batch_idx) -> dict:  # type: ignore  # pylint: disable=arguments-differ
        """To be implemented in the subclasses."""
        raise NotImplementedError

    def predict_step(self, batch, batch_idx, _):  # pylint: disable=arguments-differ, signature-differs
        """Step function called during :meth:`~pytorch_lightning.trainer.trainer.Trainer.predict`.

        By default, it calls :meth:`~pytorch_lightning.core.lightning.LightningModule.forward`.
        Override to add any processing logic.

        Args:
            batch (Tensor): Current batch
            batch_idx (int): Index of current batch
            dataloader_idx (int): Index of the current dataloader

        Return:
            Predicted output
        """
        outputs = self.validation_step(batch, batch_idx)
        self._post_process(outputs)
        outputs["pred_labels"] = outputs["pred_scores"] >= self.image_threshold.value
        return outputs

    def test_step(self, batch, _):  # pylint: disable=arguments-differ
        """Calls validation_step for anomaly map/score calculation.

        Args:
          batch (Tensor): Input batch
          _: Index of the batch.

        Returns:
          Dictionary containing images, features, true labels and masks.
          These are required in `validation_epoch_end` for feature concatenation.
        """
        return self.validation_step(batch, _)

    def validation_step_end(self, val_step_outputs):  # pylint: disable=arguments-differ
        """Called at the end of each validation step."""
        self._outputs_to_cpu(val_step_outputs)
        self._post_process(val_step_outputs)
        return val_step_outputs

    def test_step_end(self, test_step_outputs):  # pylint: disable=arguments-differ
        """Called at the end of each test step."""
        self._outputs_to_cpu(test_step_outputs)
        self._post_process(test_step_outputs)
        return test_step_outputs

    def validation_epoch_end(self, outputs):
        """Compute threshold and performance metrics.

        Args:
          outputs: Batch of outputs from the validation step
        """
        if self.hparams.model.threshold.adaptive:
            self._compute_adaptive_threshold(outputs)
        self._collect_outputs(self.image_metrics, self.pixel_metrics, outputs)
        self._log_metrics()

    def test_epoch_end(self, outputs):
        """Compute and save anomaly scores of the test set.

        Args:
            outputs: Batch of outputs from the validation step
        """
        self._collect_outputs(self.image_metrics, self.pixel_metrics, outputs)
        self._log_metrics()

    def _compute_adaptive_threshold(self, outputs):
        self._collect_outputs(self.image_threshold, self.pixel_threshold, outputs)
        self.image_threshold.compute()
        if "mask" in outputs[0].keys() and "anomaly_maps" in outputs[0].keys():
            self.pixel_threshold.compute()
        else:
            self.pixel_threshold.value = self.image_threshold.value

        self.image_metrics.F1.threshold = self.image_threshold.value.item()
        self.pixel_metrics.F1.threshold = self.pixel_threshold.value.item()

    def _collect_outputs(self, image_metric, pixel_metric, outputs):
        for output in outputs:
            image_metric.cpu()
            image_metric.update(output["pred_scores"], output["label"].int())
            if "mask" in output.keys() and "anomaly_maps" in output.keys():
                pixel_metric.cpu()
                pixel_metric.update(output["anomaly_maps"].flatten(), output["mask"].flatten().int())

    def _post_process(self, outputs):
        """Compute labels based on model predictions."""
        if "pred_scores" not in outputs and "anomaly_maps" in outputs:
            outputs["pred_scores"] = (
                outputs["anomaly_maps"].reshape(outputs["anomaly_maps"].shape[0], -1).max(dim=1).values
            )

    def _outputs_to_cpu(self, output):
        # for output in outputs:
        for key, value in output.items():
            if isinstance(value, Tensor):
                output[key] = value.cpu()

    def _log_metrics(self):
        """Log computed performance metrics."""
        self.log_dict(self.image_metrics)
        if self.hparams.dataset.task == "segmentation":
            self.log_dict(self.pixel_metrics)
