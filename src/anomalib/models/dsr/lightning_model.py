"""DSR – A Dual Subspace Re-Projection Network for Surface Anomaly Detection

Paper https://link.springer.com/chapter/10.1007/978-3-031-19821-2_31
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from os.path import isfile

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from anomalib.data.utils.augmenter import Augmenter
from anomalib.models.components import AnomalyModule
from anomalib.models.dsr.anomaly_generator import DsrAnomalyGenerator
from anomalib.models.dsr.download_weights import DsrWeightDownloader
from anomalib.models.dsr.loss import DsrSecondLoss, DsrThirdLoss
from anomalib.models.dsr.torch_model import DsrModel

__all__ = ["Dsr", "DsrLightning"]

logger = logging.getLogger(__name__)


class Dsr(AnomalyModule):
    """DSR: A Dual Subspace Re-Projection Network for Surface Anomaly Detection

    Args:
        ckpt (str): Path to checkpoint file containing the pretrained weights for the discrete
        model.
        anom_par (float, optional): Parameter determining the strength of the generated anomalies.
    """

    def __init__(self, ckpt: str, anom_par: float = 0.2) -> None:
        super().__init__()

        self.quantized_anomaly_generator = DsrAnomalyGenerator()
        self.perlin_generator = Augmenter()
        self.model = DsrModel(anom_par)
        self.second_loss = DsrSecondLoss()
        self.third_loss = DsrThirdLoss()
        self.downloader = DsrWeightDownloader()
        self.anom_par: float = anom_par
        self.ckpt = ckpt

        if not isfile("src/anomalib/models/dsr/vq_model_pretrained_128_4096.pckl"):
            logger.info("Pretrained weights not found.")
            self.downloader.download()
        else:
            logger.info("Pretrained checkpoint file found.")

    def configure_optimizers(
        self,
    ) -> tuple[dict[str, torch.optim.Optimizer | torch.optim.LRScheduler], dict[str, torch.optim.Optimizer]]:
        """Configure the Adam optimizer for training phases 2 and 3. Does not train the discrete
        model (phase 1)

        Returns:
            dict[str, torch.optim.Optimizer | torch.optim.LRScheduler]: Dictionary of optimizers
        (the first one having a schedule)
        """
        num_steps = max(
            self.trainer.max_steps // len(self.trainer.datamodule.train_dataloader()), self.trainer.max_epochs
        )
        self.second_phase = int(10 * num_steps / 12)
        anneal = int(0.8 * self.second_phase)
        optimizer_d = torch.optim.Adam(
            params=list(self.model.image_reconstruction_network.parameters())
            + list(self.model.subspace_restriction_module_hi.parameters())
            + list(self.model.subspace_restriction_module_lo.parameters())
            + list(self.model.anomaly_detection_module.parameters()),
            lr=self.hparams.model.lr,
        )
        scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, anneal, gamma=0.1)

        optimizer_u = torch.optim.Adam(params=self.model.upsampling_module.parameters(), lr=self.hparams.model.lr)

        return ({"optimizer": optimizer_d, "lr_scheduler": scheduler_d}, {"optimizer": optimizer_u})

    def on_train_start(self) -> None:
        """Load pretrained weights of the discrete model when starting training."""
        self.model.load_pretrained_discrete_model_weights(self.ckpt)

    def on_train_epoch_start(self) -> None:
        """Display a message when starting to train the upsampling module."""
        if self.current_epoch == self.second_phase:
            logger.info("Now training upsampling module.")

    def training_step(
        self, batch: dict[str, str | Tensor], batch_idx: int, optimizer_idx: int, *args, **kwargs
    ) -> STEP_OUTPUT:
        """Training Step of DSR.

        Feeds the original image and the simulated anomaly mask during first phase. During
        second phase, feeds a generated anomalous image to train the upsampling module.

        Args:
            batch (dict[str, str | Tensor]): Batch containing image filename, image, label and mask

        Returns:
            STEP_OUTPUT: Loss dictionary
        """
        del batch_idx, args, kwargs  # These variables are not used.

        if self.current_epoch < self.second_phase:
            # we are not yet training the upsampling module
            if optimizer_idx == 0:
                # we are only using the first optimizer
                input_image = batch["image"]
                # Create anomaly masks
                anomaly_mask = self.quantized_anomaly_generator.augment_batch(input_image)
                # Generate model prediction
                recon_nq_hi, recon_nq_lo, qu_hi, qu_lo, gen_img, seg, anomaly_mask = self.model(
                    input_image, anomaly_mask
                )
                # Compute loss
                loss = self.second_loss(recon_nq_hi, recon_nq_lo, qu_hi, qu_lo, input_image, gen_img, seg, anomaly_mask)
            elif optimizer_idx == 1:
                # we are not training the upsampling module
                return
            else:
                raise Exception(f"Unknown optimizer_idx {optimizer_idx}.")
        else:
            if optimizer_idx == 0:
                # we are not training the anomaly detection and object specific modules
                return
            elif optimizer_idx == 1:
                # we are training the upsampling module
                input_image = batch["image"]
                # Generate anomalies
                input_image, anomaly_maps = self.perlin_generator.augment_batch(input_image)
                # Get model prediction
                gen_masks = self.model(input_image)
                # Calculate loss
                loss = self.third_loss(gen_masks, anomaly_maps)
            else:
                raise Exception(f"Unknown optimizer_idx {optimizer_idx}.")

        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Validation step of DSR. The Softmax predictions of the anomalous class are used as anomaly map.

        Args:
            batch (dict[str, str | Tensor]): Batch of input images

        Returns:
            STEP_OUTPUT: Dictionary to which predicted anomaly maps have been added.
        """
        del args, kwargs  # These variables are not used.

        prediction, anomaly_scores = self.model(batch["image"])
        batch["anomaly_maps"] = prediction
        batch["pred_scores"] = anomaly_scores
        return batch


class DsrLightning(Dsr):
    """DSR: A Dual Subspace Re-Projection Network for Surface Anomaly Detection

    Args:
        hparams (DictConfig | ListConfig): Model parameters
    """

    def __init__(self, hparams: DictConfig | ListConfig) -> None:
        super().__init__(ckpt=hparams.model.ckpt_path, anom_par=hparams.model.anom_par)
        self.hparams: DictConfig | ListConfig  # type: ignore
        self.save_hyperparameters(hparams)
