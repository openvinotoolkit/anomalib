from typing import Any, Optional

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from anomalib.datasets.utils import Tiler


class TilingCallback(Callback):
    """
    Callback that tiles and untiles images to help detect small defects. Tiles batches of images during training and
    validation, before the batch is presented as input to the model. Untiles images and model predictions after the
    forward pass is completed.
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
        outputs["anomaly_maps"] = self.tiler.untile_batch(outputs["anomaly_maps"].unsqueeze(1))[:, 0, :, :]
        outputs["images"] = self.tiler.untile_batch(outputs["images"])

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
        outputs["anomaly_maps"] = self.tiler.untile_batch(outputs["anomaly_maps"].unsqueeze(1))[:, 0, :, :]
        outputs["images"] = self.tiler.untile_batch(outputs["images"])