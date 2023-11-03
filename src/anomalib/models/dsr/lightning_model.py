"""DSR – A Dual Subspace Re-Projection Network for Surface Anomaly Detection

Paper https://link.springer.com/chapter/10.1007/978-3-031-19821-2_31
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from anomalib.data.utils import DownloadInfo, download_and_extract
from anomalib.data.utils.augmenter import Augmenter
from anomalib.models.components import AnomalyModule
from anomalib.models.dsr.anomaly_generator import DsrAnomalyGenerator
from anomalib.models.dsr.loss import DsrSecondStageLoss, DsrThirdStageLoss
from anomalib.models.dsr.torch_model import DsrModel

__all__ = ["Dsr", "DsrLightning"]

logger = logging.getLogger(__name__)

WEIGHTS_DOWNLOAD_INFO = DownloadInfo(
    name="vq_model_pretrained_128_4096.pckl",
    url="https://github.com/openvinotoolkit/anomalib/releases/download/dsr_pretrained_weights/dsr_vq_model_pretrained.zip",
    hash="927f6b40841a7c885d12217c922b2bba",
)


class Dsr(AnomalyModule):
    """DSR: A Dual Subspace Re-Projection Network for Surface Anomaly Detection

    Args:
        latent_anomaly_strength (float, optional): Strength of the generated anomalies in the latent space.
    """

    def __init__(self, latent_anomaly_strength: float = 0.2) -> None:
        super().__init__()

        self.automatic_optimization = False

        self.quantized_anomaly_generator = DsrAnomalyGenerator()
        self.perlin_generator = Augmenter()
        self.model = DsrModel(latent_anomaly_strength)
        self.second_stage_loss = DsrSecondStageLoss()
        self.third_stage_loss = DsrThirdStageLoss()

        self.second_phase: int

    def prepare_pretrained_model(self) -> Path:
        pretrained_models_dir = Path("./pre_trained/")
        if not (pretrained_models_dir / "vq_model_pretrained_128_4096.pckl").is_file():
            download_and_extract(pretrained_models_dir, WEIGHTS_DOWNLOAD_INFO)
        return pretrained_models_dir / "vq_model_pretrained_128_4096.pckl"

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
        self.second_phase = int(num_steps * self.hparams.model.upsampling_train_ratio)
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
        ckpt: Path = self.prepare_pretrained_model()
        self.model.load_pretrained_discrete_model_weights(ckpt)

    def on_train_epoch_start(self) -> None:
        """Display a message when starting to train the upsampling module."""
        if self.current_epoch == self.second_phase:
            logger.info("Now training upsampling module.")

    def training_step(self, batch: dict[str, str | Tensor]) -> STEP_OUTPUT:
        """Training Step of DSR.

        Feeds the original image and the simulated anomaly mask during first phase. During
        second phase, feeds a generated anomalous image to train the upsampling module.

        Args:
            batch (dict[str, str | Tensor]): Batch containing image filename, image, label and mask

        Returns:
            STEP_OUTPUT: Loss dictionary
        """
        ph1_opt, ph2_opt = self.optimizers()

        if self.current_epoch < self.second_phase:
            # we are not yet training the upsampling module: we are only using the first optimizer
            input_image = batch["image"]
            # Create anomaly masks
            anomaly_mask = self.quantized_anomaly_generator.augment_batch(input_image)
            # Generate model prediction
            model_outputs = self.model(input_image, anomaly_mask)
            # Compute loss
            loss = self.second_stage_loss(
                model_outputs["recon_feat_hi"],
                model_outputs["recon_feat_lo"],
                model_outputs["embedding_bot"],
                model_outputs["embedding_top"],
                input_image,
                model_outputs["obj_spec_image"],
                model_outputs["anomaly_map"],
                model_outputs["true_anomaly_map"],
            )

            # compute manual optimizer step
            ph1_opt.zero_grad()
            self.manual_backward(loss)
            ph1_opt.step()

        else:
            # we are training the upsampling module
            input_image = batch["image"]
            # Generate anomalies
            input_image, anomaly_maps = self.perlin_generator.augment_batch(input_image)
            # Get model prediction
            model_outputs = self.model(input_image)
            # Calculate loss
            loss = self.third_stage_loss(model_outputs["anomaly_map"], anomaly_maps)

            # compute manual optimizer step
            ph2_opt.zero_grad()
            self.manual_backward(loss)
            ph2_opt.step()

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

        model_outputs = self.model(batch["image"])
        batch["anomaly_maps"] = model_outputs["anomaly_map"]
        batch["pred_scores"] = model_outputs["pred_score"]
        return batch


class DsrLightning(Dsr):
    """DSR: A Dual Subspace Re-Projection Network for Surface Anomaly Detection

    Args:
        hparams (DictConfig | ListConfig): Model parameters
    """

    def __init__(self, hparams: DictConfig | ListConfig) -> None:
        super().__init__(latent_anomaly_strength=hparams.model.latent_anomaly_strength)
        self.hparams: DictConfig | ListConfig  # type: ignore
        self.save_hyperparameters(hparams)
