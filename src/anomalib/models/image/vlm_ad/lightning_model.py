"""Visual Anomaly Model for Zero/Few-Shot Anomaly Classification."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from enum import Enum

import torch
from torch.utils.data import DataLoader

from anomalib import LearningType
from anomalib.models import AnomalyModule

from .backends import Backend, ChatGPT, Ollama

logger = logging.getLogger(__name__)


class VlmAdBackend(Enum):
    """Supported VLM backends."""

    OLLAMA = "ollama"
    CHATGPT = "chatgpt"


class VlmAd(AnomalyModule):
    """Visual anomaly model."""

    def __init__(
        self,
        backend: VlmAdBackend | str = VlmAdBackend.OLLAMA,
        api_key: str | None = None,
        k_shot: int = 3,
    ) -> None:
        super().__init__()
        self.k_shot = k_shot
        backend = VlmAdBackend(backend)
        self.vlm_backend: Backend = self._setup_vlm(backend, api_key)

    @staticmethod
    def _setup_vlm(backend: VlmAdBackend, api_key: str | None) -> Backend:
        match backend:
            case VlmAdBackend.OLLAMA:
                return Ollama()
            case VlmAdBackend.CHATGPT:
                return ChatGPT(api_key=api_key)
            case _:
                msg = f"Unsupported VLM backend: {backend}"
                raise ValueError(msg)

    def _setup(self) -> None:
        if self.k_shot:
            logger.info("Collecting reference images from training dataset.")
            dataloader = self.trainer.datamodule.train_dataloader()
            self.collect_reference_images(dataloader)

    def collect_reference_images(self, dataloader: DataLoader) -> None:
        """Collect reference images for few-shot inference."""
        count = 0
        for batch in dataloader:
            for img_path in batch["image_path"]:
                self.vlm_backend.add_reference_images(img_path)
                count += 1
                if count == self.k_shot:
                    return

    def validation_step(
        self,
        batch: dict[str, str | torch.Tensor],
        *args,
        **kwargs,
    ) -> dict:
        """Validation step."""
        del args, kwargs  # These variables are not used.
        responses = [(self.vlm_backend.predict(img_path)) for img_path in batch["image_path"]]

        batch["str_output"] = responses
        batch["pred_scores"] = torch.tensor(
            [1.0 if r.startswith("Y") else 0.0 for r in responses],
            device=self.device,
        )
        return batch

    @property
    def learning_type(self) -> LearningType:
        """The learning type of the model."""
        return LearningType.ZERO_SHOT if self.k_shot == 0 else LearningType.FEW_SHOT

    @property
    def trainer_arguments(self) -> dict[str, int | float]:
        """Doesn't need training."""
        return {}

    @staticmethod
    def configure_transforms(image_size: tuple[int, int] | None = None) -> None:
        """This modes does not require any transforms."""
        if image_size is not None:
            logger.warning(
                "Ignoring image_size argument as each backend has its own transforms.",
            )
