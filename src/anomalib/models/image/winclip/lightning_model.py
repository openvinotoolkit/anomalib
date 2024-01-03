"""WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation.

Paper https://arxiv.org/abs/2303.14814
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT  # noqa: TCH002
from torch import Tensor

from anomalib.models.components import AnomalyModule
from anomalib.models.image.winclip.torch_model import WinClipModel

logger = logging.getLogger(__name__)

__all__ = ["WinClip"]


class WinClip(AnomalyModule):
    """WinCLIP Lightning model.

    Args:
        class_name (str): The name of the object class used in the prompt ensemble.
            Defaults to ``object``.
        k_shot (int): The number of reference images for few-shot inference.
            Defaults to ``0``.
    """

    def __init__(
        self,
        class_name: str | None = None,
        k_shot: int = 0,
        scales: tuple = (2, 3),
    ) -> None:
        super().__init__()
        self.model = WinClipModel(k_shot=k_shot, scales=scales)
        self.class_name = class_name
        self.k_shot = k_shot

    def setup(self, stage: str) -> None:
        """Setup WinCLIP.

        - Set the class name used in the prompt ensemble.
        - Prepare the mask locations for the sliding-window approach.
        - Collect text embeddings for zero-shot inference.
        - Collect reference images for few-shot inference.

        We need to pass the device because this hook is called before the model is moved to the device.
        """
        del stage
        self._set_class_name()
        self.model.prepare_masks(device=self.device)
        self.model.collect_text_embeddings(self.class_name, device=self.device)

        if self.k_shot:
            ref_images = self.collect_reference_images()
            self.model.collect_visual_embeddings(ref_images, device=self.device)

    def _set_class_name(self) -> None:
        """Set the class name used in the prompt ensemble.

        - When a class name is provided by the user, it is used.
        - When the user did not provide a class name, the category name from the datamodule is used, if available.
        - When the user did not provide a class name and the datamodule does not have a category name, the default
            class name "object" is used.
        """
        if self.class_name is not None:
            logger.info("Using class name: %s", self.class_name)
            return
        if hasattr(self.trainer.datamodule, "category"):
            logger.info("No class name provided, using category from datamodule: %s", self.trainer.datamodule.category)
            self.class_name = self.trainer.datamodule.category
        else:
            logger.info("No class name provided and no category name found in datamodule using default: object")
            self.class_name = "object"

    def collect_reference_images(self) -> torch.Tensor:
        """Collect reference images for few-shot inference.

        The reference images are collected by iterating the training dataset until the required number of images are
        collected.

        Returns:
            ref_images (Tensor): A tensor containing the reference images.
        """
        ref_images = Tensor()
        for batch in self.trainer.datamodule.train_dataloader():
            images = batch["image"][: self.k_shot - ref_images.shape[0]]
            ref_images = torch.cat((ref_images, images))
            if self.k_shot == ref_images.shape[0]:
                break
        return ref_images

    @staticmethod
    def configure_optimizers() -> None:  # pylint: disable=arguments-differ
        """WinCLIP doesn't require optimization, therefore returns no optimizers."""
        return

    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> None:
        """Training Step of WinCLIP."""
        del batch, args, kwargs

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Validation Step of WinCLIP."""
        del args, kwargs  # These variables are not used.
        batch["pred_scores"], batch["anomaly_maps"] = self.model(batch["image"])
        return batch

    @property
    def trainer_arguments(self) -> dict[str, int | float]:
        """Set model-specific trainer arguments."""
        return {
            "max_epochs": 1,
            "limit_train_batches": 1,
        }
