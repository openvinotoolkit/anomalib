"""Anomaly Score Normalization Callback."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.distributions import LogNormal

from anomalib.models import get_model
from anomalib.models.components import AnomalyModule
from anomalib.post_processing.normalization.cdf import normalize, standardize
from anomalib.utils.metrics import AnomalyScoreDistribution

logger = logging.getLogger(__name__)


class CdfNormalizationCallback(Callback):
    """Callback that standardizes the image-level and pixel-level anomaly scores."""

    def __init__(self) -> None:
        self.image_dist: LogNormal | None = None
        self.pixel_dist: LogNormal | None = None

    def setup(self, trainer: pl.Trainer, pl_module: AnomalyModule, stage: str | None = None) -> None:
        """Adds training_distribution metrics to normalization metrics."""
        del trainer, stage  # These variabels are not used.

        if not hasattr(pl_module, "normalization_metrics"):
            pl_module.normalization_metrics = AnomalyScoreDistribution().cpu()
        elif not isinstance(pl_module.normalization_metrics, AnomalyScoreDistribution):
            raise AttributeError(
                f"Expected normalization_metrics to be of type AnomalyScoreDistribution,"
                f" got {type(pl_module.normalization_metrics)}"
            )

    def on_test_start(self, trainer: pl.Trainer, pl_module: AnomalyModule) -> None:
        """Called when the test begins."""
        del trainer  # `trainer` variable is not used.

        if pl_module.image_metrics is not None:
            pl_module.image_metrics.set_threshold(0.5)
        if pl_module.pixel_metrics is not None:
            pl_module.pixel_metrics.set_threshold(0.5)

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: AnomalyModule) -> None:
        """Called when the validation starts after training.

        Use the current model to compute the anomaly score distributions
        of the normal training data. This is needed after every epoch, because the statistics must be
        stored in the state dict of the checkpoint file.
        """
        logger.info("Collecting the statistics of the normal training data to normalize the scores.")
        self._collect_stats(trainer, pl_module)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the validation batch ends, standardizes the predicted scores and anomaly maps."""
        del trainer, batch, batch_idx, dataloader_idx  # These variables are not used.

        self._standardize_batch(outputs, pl_module)

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the test batch ends, normalizes the predicted scores and anomaly maps."""
        del trainer, batch, batch_idx, dataloader_idx  # These variables are not used.

        self._standardize_batch(outputs, pl_module)
        self._normalize_batch(outputs, pl_module)

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: AnomalyModule,
        outputs: dict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the predict batch ends, normalizes the predicted scores and anomaly maps."""
        del trainer, batch, batch_idx, dataloader_idx  # These variables are not used.

        self._standardize_batch(outputs, pl_module)
        self._normalize_batch(outputs, pl_module)
        outputs["pred_labels"] = outputs["pred_scores"] >= 0.5

    def _collect_stats(self, trainer: pl.Trainer, pl_module: AnomalyModule) -> None:
        """Collect the statistics of the normal training data.

        Create a trainer and use it to predict the anomaly maps and scores of the normal training data. Then
         estimate the distribution of anomaly scores for normal data at the image and pixel level by computing
         the mean and standard deviations. A dictionary containing the computed statistics is stored in self.stats.
        """
        predictions = Trainer(accelerator=trainer.accelerator, devices=trainer.num_devices).predict(
            model=self._create_inference_model(pl_module), dataloaders=trainer.datamodule.train_dataloader()
        )
        pl_module.normalization_metrics.reset()
        for batch in predictions:
            if "pred_scores" in batch.keys():
                pl_module.normalization_metrics.update(anomaly_scores=batch["pred_scores"])
            if "anomaly_maps" in batch.keys():
                pl_module.normalization_metrics.update(anomaly_maps=batch["anomaly_maps"])
        pl_module.normalization_metrics.compute()

    @staticmethod
    def _create_inference_model(pl_module):
        """Create a duplicate of the PL module that can be used to perform inference on the training set."""
        new_model = get_model(pl_module.hparams)
        new_model.normalization_metrics = AnomalyScoreDistribution().cpu()
        new_model.load_state_dict(pl_module.state_dict())
        return new_model

    @staticmethod
    def _standardize_batch(outputs: STEP_OUTPUT, pl_module) -> None:
        stats = pl_module.normalization_metrics.to(outputs["pred_scores"].device)
        outputs["pred_scores"] = standardize(outputs["pred_scores"], stats.image_mean, stats.image_std)
        if "anomaly_maps" in outputs.keys():
            outputs["anomaly_maps"] = standardize(
                outputs["anomaly_maps"], stats.pixel_mean, stats.pixel_std, center_at=stats.image_mean
            )

    @staticmethod
    def _normalize_batch(outputs: STEP_OUTPUT, pl_module: AnomalyModule) -> None:
        outputs["pred_scores"] = normalize(outputs["pred_scores"], pl_module.image_threshold.value)
        if "anomaly_maps" in outputs.keys():
            outputs["anomaly_maps"] = normalize(outputs["anomaly_maps"], pl_module.pixel_threshold.value)
            outputs["anomaly_maps"] = normalize(outputs["anomaly_maps"], pl_module.pixel_threshold.value)
