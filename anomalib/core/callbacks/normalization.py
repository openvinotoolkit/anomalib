"""Anomaly Score Normalization Callback."""
import copy
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback, Trainer
from torch.distributions import LogNormal, Normal

from anomalib.core.metrics.training_stats import TrainingStats


class NormalizationCallback(Callback):
    """Callback that standardizes the image-level and pixel-level anomaly scores."""

    def __init__(self):
        self.image_dist: Optional[LogNormal] = None
        self.pixel_dist: Optional[LogNormal] = None

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, _unused: Optional[Any] = None
    ) -> None:
        """Called when the train epoch ends.

        Use the current model to compute the anomaly score distributions
        of the normal training data. This is needed after every epoch, because the statistics must be
        stored in the state dict of the checkpoint file.
        """
        self._collect_stats(trainer, pl_module)

    def on_validation_batch_end(
        self,
        _trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Dict,
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Called when the validation batch ends, standardizes the predicted scores and anomaly maps."""
        self._standardize(outputs, pl_module)

    def on_test_batch_end(
        self,
        _trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict,
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Called when the test batch ends, normalizes the predicted scores and anomaly maps."""
        self._standardize(outputs, pl_module)
        self._normalize(outputs, pl_module)

    def on_predict_batch_end(
        self,
        _trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict,
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Called when the predict batch ends, normalizes the predicted scores and anomaly maps."""
        self._standardize(outputs, pl_module)
        self._normalize(outputs, pl_module)

    def _collect_stats(self, trainer, pl_module):
        """Collect the statistics of the normal training data.

        Create a trainer and use it to predict the anomaly maps and scores of the normal training data. Then
         estimate the distribution of anomaly scores for normal data at the image and pixel level by computing
         the mean and standard deviations. A dictionary containing the computed statistics is stored in self.stats.
        """
        predictions = Trainer(gpus=trainer.gpus).predict(
            model=copy.deepcopy(pl_module), dataloaders=trainer.datamodule.train_dataloader()
        )
        training_stats = TrainingStats()
        for batch in predictions:
            if "pred_scores" in batch.keys():
                training_stats.update(anomaly_scores=batch["pred_scores"])
            if "anomaly_maps" in batch.keys():
                training_stats.update(anomaly_maps=batch["anomaly_maps"])
        stats = training_stats()
        pl_module.image_mean = stats["image_mean"]
        pl_module.image_std = stats["image_std"]
        pl_module.pixel_mean = stats["pixel_mean"]
        pl_module.pixel_std = stats["pixel_std"]

    def _standardize(self, outputs: Dict, pl_module) -> None:
        """Standardize the predicted scores and anomaly maps to the z-domain."""
        device = outputs["pred_scores"].device
        outputs["pred_scores"] = (torch.log(outputs["pred_scores"]) - pl_module.image_mean) / pl_module.image_std
        if "anomaly_maps" in outputs.keys():
            outputs["anomaly_maps"] = (
                torch.log(outputs["anomaly_maps"]) - pl_module.pixel_mean.to(device)
            ) / pl_module.pixel_std.to(device)
            outputs["anomaly_maps"] -= (
                pl_module.image_mean.to(device) - pl_module.pixel_mean.to(device)
            ) / pl_module.pixel_std.to(device)

    def _normalize(self, outputs: Dict, pl_module: pl.LightningModule) -> None:
        """Normalize the predicted scores and anomaly maps by first standardizing and then computing the CDF."""
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
