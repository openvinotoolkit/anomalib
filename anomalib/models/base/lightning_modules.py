"""
Base Anomaly Lightning Models
"""

from pathlib import Path
from typing import List, Union

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from pytorch_lightning.callbacks.base import Callback
from sklearn.metrics import roc_auc_score
from torch import Tensor

from anomalib.core.utils.anomaly_map_generator import BaseAnomalyMapGenerator
from anomalib.models.base.torch_modules import BaseAnomalyModule


class BaseAnomalyLightning(pl.LightningModule):
    """
    BaseAnomalyModel
    """

    def __init__(self, params: Union[DictConfig, ListConfig]):
        super().__init__()
        # Force the type for hparams so that it works with OmegaConfig style of accessing
        self.hparams: Union[DictConfig, ListConfig]  # type: ignore
        self.save_hyperparameters(params)
        self.supported_tasks: List[str]
        self.loss: torch.Tensor
        self.callbacks: List[Callback]

        self.filenames: List[Union[str, Path]]
        self.images: List[Union[np.ndarray, Tensor]]

        self.true_masks: List[Union[np.ndarray, Tensor]]
        self.anomaly_maps: List[Union[np.ndarray, Tensor]]

        self.true_labels: List[Union[np.ndarray, Tensor]]
        self.pred_labels: List[Union[np.ndarray, Tensor]]

        self.image_roc_auc: float
        self.pixel_roc_auc: float

        self.image_f1_score: float

        self.model: BaseAnomalyModule

    def check_task_support(self, task):
        """
        Check if the task type is supported by the algorithm, if not stop with error.
        """
        if task not in self.supported_tasks:
            raise ValueError(
                f"{task} is not supported by {self.hparams.model.name} model. "
                f"Please change task type in config file to one of {self.supported_tasks}"
            )

    def test_step(self, batch, _):
        """Calls validation_step for anomaly map/score calculation.

        Args:
          batch: Input batch
          _: Index of the batch.

        Returns:
          Dictionary containing images, features, true labels and masks.
          These are required in `validation_epoch_end` for feature concatenation.

        """
        return self.validation_step(batch, _)

    def validation_epoch_end(self, outputs):
        """Compute image and pixel level roc scores depending on task type

        Args:
          outputs: Batch of outputs from the validation step

        Returns:

        """

        self.filenames = [Path(f) for x in outputs for f in x["image_path"]]
        self.images = torch.vstack([x["image"] for x in outputs])

        self.true_labels = np.hstack([output["label"].cpu() for output in outputs])
        if "pred_labels" not in outputs[0] and "anomaly_maps" in outputs[0]:
            self.anomaly_maps = np.vstack([output["anomaly_maps"].cpu() for output in outputs])
            self.pred_labels = self.anomaly_maps.reshape(self.anomaly_maps.shape[0], -1).max(axis=1)
        else:
            self.pred_labels = np.hstack([output["pred_labels"].cpu() for output in outputs])
        self.image_roc_auc = roc_auc_score(self.true_labels, self.pred_labels)

        _, self.image_f1_score = BaseAnomalyMapGenerator.compute_adaptive_threshold(self.true_labels, self.pred_labels)

        self.log(
            name="Image-Level AUC",
            value=self.image_roc_auc,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            name="Image-Level F1",
            value=self.image_f1_score,
            on_epoch=True,
            prog_bar=True,
        )

        if self.hparams.dataset.task == "segmentation":
            self.true_masks = np.vstack([output["mask"].squeeze(1).cpu() for output in outputs])
            self.anomaly_maps = np.vstack([output["anomaly_maps"].cpu() for output in outputs])
            self.pixel_roc_auc = roc_auc_score(self.true_masks.flatten(), self.anomaly_maps.flatten())

            self.log(
                name="Pixel-Level AUC",
                value=self.pixel_roc_auc,
                on_epoch=True,
                prog_bar=True,
            )

    def test_epoch_end(self, outputs):
        """
        Compute and save anomaly scores of the test set.

        Args:
            outputs: Batch of outputs from the validation step

        """
        self.validation_epoch_end(outputs)
