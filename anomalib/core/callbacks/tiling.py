from typing import Any, Optional

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from anomalib.datasets.utils import Tiler


class TilingCallback(Callback):
    r"""
    Abstract base class used to build new callbacks.

    Subclass this class and override any of the relevant hooks
    """

    def __init__(self, hparams):
        self.tiler = Tiler(hparams.dataset.tile_size)

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the train batch begins."""
        batch["image"] = self.tiler.tile_batch(batch["image"])

    def on_train_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if "features" in outputs.keys():
            for feature_name in outputs["features"]:
                outputs["features"][feature_name] = self.tiler.untile_batch(outputs["features"][feature_name])

    def on_validation_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the validation batch begins."""
        batch["image"] = self.tiler.tile_batch(batch["image"])

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the validation batch ends."""
        if "anomaly_maps" in outputs.keys():
            outputs["anomaly_maps"] = self.tiler.untile_batch(outputs["anomaly_maps"].unsqueeze(1))[:, 0, :, :]
        if "images" in outputs.keys():
            outputs["images"] = self.tiler.untile_image(outputs["images"])
        if "features" in outputs.keys():
            for feature_name in outputs["features"]:
                outputs["features"][feature_name] = self.tiler.untile_batch(outputs["features"][feature_name])

    def on_test_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the test batch begins."""
        batch["image"] = self.tiler.tile_batch(batch["image"])

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the test batch ends."""
        if "anomaly_maps" in outputs.keys():
            outputs["anomaly_maps"] = self.tiler.untile_batch(outputs["anomaly_maps"].unsqueeze(1))[:, 0, :, :]
        if "images" in outputs.keys():
            outputs["images"] = self.tiler.untile_image(outputs["images"])
        if "features" in outputs.keys():
            for feature_name in outputs["features"]:
                outputs["features"][feature_name] = self.tiler.untile_batch(outputs["features"][feature_name])