"""
Base Anomaly Module for Training Task
"""

from pathlib import Path
from typing import List, Union

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.callbacks.base import Callback
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch import nn

from anomalib.core.results import ClassificationResults, SegmentationResults
from anomalib.utils.metrics import compute_threshold_and_f1_score


class AnomalyModule(pl.LightningModule):
    """
    AnomalyModule to train, validate, predict and test images.

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

        self.model: nn.Module

        self.results: Union[ClassificationResults, SegmentationResults]
        if params.dataset.task == "classification":
            self.results = ClassificationResults()
        elif params.dataset.task == "segmentation":
            self.results = SegmentationResults()
        else:
            raise NotImplementedError("Only Classification and Segmentation tasks are supported in this version.")

    def forward(self, x):  # pylint: disable=arguments-differ
        """
        Forward-pass input tensor to the module

        Args:
            x (Tensor): Input Tensor

        Returns:
            [Tensor]: Output tensor from the model.
        """
        return self.model(x)

    def test_step(self, batch, _):  # pylint: disable=arguments-differ
        """
        Calls validation_step for anomaly map/score calculation.

        Args:
          batch: Input batch
          _: Index of the batch.

        Returns:
          Dictionary containing images, features, true labels and masks.
          These are required in `validation_epoch_end` for feature concatenation.

        """
        return self.validation_step(batch, _)

    def validation_epoch_end(self, outputs):
        """
        Compute image-level performance metrics

        Args:
          outputs: Batch of outputs from the validation step


        """

        self.results.filenames = [Path(f) for x in outputs for f in x["image_path"]]
        self.results.images = torch.vstack([x["image"] for x in outputs])

        self.results.true_labels = np.hstack([output["label"].cpu() for output in outputs])

        if "pred_scores" not in outputs[0] and "anomaly_maps" in outputs[0]:
            self.results.anomaly_maps = np.vstack([output["anomaly_maps"].cpu() for output in outputs])
            self.results.pred_scores = self.results.anomaly_maps.reshape(self.results.anomaly_maps.shape[0], -1).max(
                axis=1
            )
        else:
            self.results.pred_scores = np.hstack([output["pred_scores"].cpu() for output in outputs])

        self.results.performance["image_roc_auc"] = roc_auc_score(self.results.true_labels, self.results.pred_scores)
        threshold_value, self.results.performance["image_f1_score"] = compute_threshold_and_f1_score(
            self.results.true_labels, self.results.pred_scores
        )

        self.results.pred_labels = self.results.pred_scores > threshold_value

        self.results.performance["balanced_accuracy_score"] = balanced_accuracy_score(
            self.results.true_labels, self.results.pred_labels
        )

        if self.hparams.dataset.task == "segmentation":
            self.results.true_masks = np.vstack([output["mask"].squeeze(1).cpu() for output in outputs])
            self.results.anomaly_maps = np.vstack([output["anomaly_maps"].cpu() for output in outputs])
            self.results.performance["pixel_roc_auc"] = roc_auc_score(
                self.results.true_masks.flatten(), self.results.anomaly_maps.flatten()
            )

        for name, value in self.results.performance.items():
            self.log(name=name, value=value, on_epoch=True, prog_bar=True)

    def test_epoch_end(self, outputs):
        """
        Compute and save anomaly scores of the test set.

        Args:
            outputs: Batch of outputs from the validation step

        """
        self.validation_epoch_end(outputs)
