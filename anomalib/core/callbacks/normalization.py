"""Anomaly Score Normalization Callback."""
import copy
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.distributions import LogNormal, Normal

from anomalib.core.metrics.training_stats import TrainingStats


class NormalizationCallback(Callback):
    """Callback that standardizes the image-level and pixel-level anomaly scores."""

    def __init__(self):
        self.image_dist: Optional[LogNormal] = None
        self.pixel_dist: Optional[LogNormal] = None
        self.stats: Dict = {}

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, unused: Optional[Any] = None
    ) -> None:
        """Called when the train epoch ends.

        Use the current model to compute the anomaly score distributions
        of the normal training data. This is needed after every epoch, because the statistics must be
        stored in the state dict of the checkpoint file.
        """
        self._collect_stats(trainer, pl_module)

    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint) -> dict:
        """Called when saving a model checkpoint, used to persist state."""
        return self.stats

    def on_load_checkpoint(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, callback_state: Dict[str, Any]
    ) -> None:
        """Called when loading a model checkpoint, use to reload state."""
        self.stats.update(callback_state)

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Caled when the validation batch ends, standardizes the predicted scores and anomaly maps."""
        self._standardize(outputs)

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the test batch ends."""
        self._standardize(outputs)
        self._normalize(outputs, pl_module)

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the predict batch ends."""
        self._standardize(outputs)
        self._normalize(outputs, pl_module)

    def _collect_stats(self, trainer, pl_module):
        predictions = Trainer(gpus=trainer.gpus).predict(
            model=copy.deepcopy(pl_module), dataloaders=trainer.datamodule.train_dataloader()
        )
        training_stats = TrainingStats()
        for batch in predictions:
            if "pred_scores" in batch.keys():
                training_stats.update(anomaly_scores=batch["pred_scores"])
            if "anomaly_maps" in batch.keys():
                training_stats.update(anomaly_maps=batch["anomaly_maps"])
        self.stats = training_stats()

    def _standardize(self, outputs):
        device = outputs["pred_scores"].device
        outputs["pred_scores"] = (torch.log(outputs["pred_scores"]) - self.stats["image_mean"]) / self.stats[
            "image_std"
        ]
        if "anomaly_maps" in outputs.keys():
            outputs["anomaly_maps"] = (
                torch.log(outputs["anomaly_maps"]) - self.stats["pixel_mean"].to(device)
            ) / self.stats["pixel_std"].to(device)

    def _normalize(self, outputs, pl_module):
        device = outputs["pred_scores"].device
        image_dist = Normal(torch.Tensor([0]), torch.Tensor([1]))
        outputs["pred_scores"] = image_dist.cdf(outputs["pred_scores"].cpu() - pl_module.image_threshold.cpu()).to(
            device
        )
        if "anomaly_maps" in outputs.keys():
            pixel_dist = Normal(torch.Tensor([0]), torch.Tensor([1]))
            outputs["anomaly_maps"] = pixel_dist.cdf(
                outputs["anomaly_maps"].cpu() - pl_module.pixel_threshold.cpu()
            ).to(device)
