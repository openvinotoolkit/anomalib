"""WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation

Paper https://arxiv.org/abs/2303.14814
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence, Union
from lightning.pytorch.callbacks.callback import Callback

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torchvision.transforms.functional import resize

from anomalib.models.components import AnomalyModule
from anomalib.models.image.winclip.torch_model import WinClipAD, WinClipModel
from anomalib.utils.callbacks.model_checkpoint_zero_shot import ZeroShotModelCheckpoint
from lightning.pytorch.callbacks import ModelCheckpoint

logger = logging.getLogger(__name__)

__all__ = ["WinClip", "WinClipLightning"]


class WinClip(AnomalyModule):
    """WinCLIP"""

    def __init__(
        self,
        class_name: str,
        n_shot: int,
    ) -> None:
        super().__init__()
        # self.model = WinClipAD()
        self.model = WinClipModel()
        self.class_name = class_name
        self.n_shot = n_shot

    def setup(self, stage) -> None:
        del stage
        self.model.collect_text_embeddings(self.class_name, device=self.device)

    def on_train_start(self) -> None:
        if self.n_shot:
            gallery = self.collect_image_gallery()
            self.model.build_image_feature_gallery(gallery)

    def collect_image_gallery(self):
        gallery = Tensor().to(self.device)
        for batch in self.trainer.datamodule.train_dataloader():
            images = batch["image"][:self.n_shot-gallery.shape[0]].to(self.device)
            gallery = torch.cat((gallery, images))
            if self.n_shot == gallery.shape[0]:
                break
        return gallery

    @staticmethod
    def configure_optimizers() -> None:  # pylint: disable=arguments-differ
        """WinCLIP doesn't require optimization, therefore returns no optimizers."""
        return None

    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> None:
        """Training Step of WinCLIP"""
        return None

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Validation Step of WinCLIP"""
        del args, kwargs  # These variables are not used.
        image_scores, pixel_scores = self.model(batch["image"])
        # scores = [resize(torch.tensor(score).unsqueeze(0), batch["image"].shape[-2:]).squeeze() for score in scores]
        # batch["anomaly_maps"] = torch.stack(scores).to(self.device)
        batch["pred_scores"] = image_scores
        batch["anomaly_maps"] = pixel_scores
        return batch

    @property
    def trainer_arguments(self) -> dict[str, int | float]:
        """Set model-specific trainer arguments."""
        return {
            "max_epochs": 1,
            "limit_train_batches": 1,
        }
