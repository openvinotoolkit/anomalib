"""DSR – A Dual Subspace Re-Projection Network for Surface Anomaly Detection

Paper https://link.springer.com/chapter/10.1007/978-3-031-19821-2_31
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Callable
from pathlib import Path

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, nn

from anomalib.models.dsr.anomaly_generator import DsrAnomalyGenerator
from anomalib.data.utils.augmenter import Augmenter
from anomalib.models.components import AnomalyModule
from anomalib.models.dsr.loss import DsrLoss
from anomalib.models.dsr.torch_model import DsrModel

__all__ = ["Dsr", "DsrLightning"]


class Dsr(AnomalyModule):
    """DSR: A Dual Subspace Re-Projection Network for Surface Anomaly Detection

    Args:
        anomaly_source_path (str | None): Path to folder that contains the anomaly source images. Random noise will
            be used if left empty.
    """

    def __init__(self, anom_par: float = 0.2) -> None:
        super().__init__()

        # while "model < objective or end epoch" on train
        # else train upsampling module till epoch end

        self.anomaly_generator = DsrAnomalyGenerator()
        self.model = DsrModel(anom_par)
        self.loss = DsrLoss()
        self.anom_par: float = anom_par


    def on_training_start(self) -> STEP_OUTPUT:
        # TODO: load weights for the discrete latent model, or do it as 'on training start'?
        pass


    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Training Step of DSR.

        Feeds the original image and the simulated anomaly mask during first phase. During
        second phase, feeds a generated anomalous image to train the upsampling module.

        Args:
            batch (dict[str, str | Tensor]): Batch containing image filename, image, label and mask

        Returns:
            Loss dictionary
        """
        del args, kwargs  # These variables are not used.

        input_image = batch["image"]
        # Create anomaly masks
        anomaly_mask = self.anomaly_generator.augment_batch(input_image)
        # Generate model prediction
        """Compute the anomaly mask from an input image.

        Args:
            batch (Tensor): Batch of input images.
        return recon_feat_hi, recon_feat_lo, embd_top, embd_bot, gen_image_def, out_mask_sm, anomaly_map
        Returns:
            Tuple of :
                - Reconstructed non-quantized features (high)
                - Reconstructed non-quantized features (low)
                - Quantized features (high)
                - Quantized features (low)
                - General object decoder-decoded image
                - Computed segmentation mask
                - Resized ground-truth anomaly map
        """
        recon_nq_hi, recon_nq_lo, qu_hi, qu_lo, gen_img, seg, anomaly_mask = self.model(input_image, anomaly_mask)
        # Compute loss
        loss = self.loss(recon_nq_hi, recon_nq_lo, qu_hi, qu_lo, input_image, gen_img, seg, anomaly_mask)

        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}
    

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Validation step of DRAEM. The Softmax predictions of the anomalous class are used as anomaly map.

        Args:
            batch (dict[str, str | Tensor]): Batch of input images

        Returns:
            Dictionary to which predicted anomaly maps have been added.
        """
        del args, kwargs  # These variables are not used.

        prediction = self.model(batch["image"])
        batch["anomaly_maps"] = prediction
        return batch


class DsrLightning(Dsr):
    """DRÆM: A discriminatively trained reconstruction embedding for surface anomaly detection.

    Args:
        hparams (DictConfig | ListConfig): Model parameters
    """

    def __init__(self, hparams: DictConfig | ListConfig) -> None:
        super().__init__(
            enable_sspcab=hparams.model.enable_sspcab,
            sspcab_lambda=hparams.model.sspcab_lambda,
            anomaly_source_path=hparams.model.anomaly_source_path,
        )
        self.hparams: DictConfig | ListConfig  # type: ignore
        self.save_hyperparameters(hparams)

    def configure_callbacks(self) -> list[EarlyStopping]:
        """Configure model-specific callbacks.

        Note:
            This method is used for the existing CLI.
            When PL CLI is introduced, configure callback method will be
                deprecated, and callbacks will be configured from either
                config.yaml file or from CLI.
        """
        early_stopping = EarlyStopping(
            monitor=self.hparams.model.early_stopping.metric,
            patience=self.hparams.model.early_stopping.patience,
            mode=self.hparams.model.early_stopping.mode,
        )
        return [early_stopping]

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the Adam optimizer."""
        return torch.optim.Adam(params=self.model.parameters(), lr=self.hparams.model.lr)
