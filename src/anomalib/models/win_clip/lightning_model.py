"""WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation

Paper https://arxiv.org/abs/2303.14814
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torchvision.transforms.functional import resize

from anomalib.models.components import AnomalyModule
from anomalib.models.win_clip.torch_model import WinClipAD

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
        self.model = WinClipAD()
        self.class_name = class_name
        self.n_shot = n_shot

    def on_fit_start(self) -> None:
        self.model.build_text_feature_gallery(self.class_name)

    @staticmethod
    def configure_optimizers() -> None:  # pylint: disable=arguments-differ
        """WinCLIP doesn't require optimization, therefore returns no optimizers."""
        return None

    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> None:
        """Training Step of WinCLIP"""
        del args, kwargs  # These variables are not used.
        if self.n_shot:
            self.model.build_image_feature_gallery(batch["image"][: self.n_shot])
        return

    def on_validation_start(self) -> None:
        pass

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Validation Step of WinCLIP"""
        del args, kwargs  # These variables are not used.
        scores = self.model(batch["image"])
        scores = [resize(torch.tensor(score).unsqueeze(0), batch["image"].shape[-2:]).squeeze() for score in scores]
        batch["anomaly_maps"] = torch.stack(scores).to(self.device)
        return batch


class WinClipLightning(WinClip):
    """WinCLIP"""

    def __init__(self, hparams: DictConfig | ListConfig) -> None:
        super().__init__(
            class_name=hparams.model.class_name,
            n_shot=hparams.model.n_shot,
        )
        self.hparams: DictConfig | ListConfig  # type: ignore
        self.save_hyperparameters(hparams)
