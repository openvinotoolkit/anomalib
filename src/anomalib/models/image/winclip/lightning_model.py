"""WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation.

Paper https://arxiv.org/abs/2303.14814
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, InterpolationMode, Normalize, Resize, Transform

from anomalib import LearningType
from anomalib.data.predict import PredictDataset
from anomalib.models.components import AnomalyModule

from .torch_model import WinClipModel

logger = logging.getLogger(__name__)

__all__ = ["WinClip"]


class WinClip(AnomalyModule):
    """WinCLIP Lightning model.

    Args:
        class_name (str, optional): The name of the object class used in the prompt ensemble.
            Defaults to ``None``.
        k_shot (int): The number of reference images for few-shot inference.
            Defaults to ``0``.
        scales (tuple[int], optional): The scales of the sliding windows used for multiscale anomaly detection.
            Defaults to ``(2, 3)``.
        few_shot_source (str | Path, optional): Path to a folder of reference images used for few-shot inference.
            Defaults to ``None``.
    """

    EXCLUDE_FROM_STATE_DICT = frozenset({"model.clip"})

    def __init__(
        self,
        class_name: str | None = None,
        k_shot: int = 0,
        scales: tuple = (2, 3),
        few_shot_source: Path | str | None = None,
    ) -> None:
        super().__init__()
        self.model = WinClipModel(scales=scales, apply_transform=False)
        self.class_name = class_name
        self.k_shot = k_shot
        self.few_shot_source = Path(few_shot_source) if few_shot_source else None

    def _setup(self) -> None:
        """Setup WinCLIP.

        - Set the class name used in the prompt ensemble.
        - Collect text embeddings for zero-shot inference.
        - Collect reference images for few-shot inference.

        We need to pass the device because this hook is called before the model is moved to the device.
        """
        # get class name
        self.class_name = self._get_class_name()
        ref_images = None

        # get reference images
        if self.k_shot:
            if self.few_shot_source:
                logger.info("Loading reference images from %s", self.few_shot_source)
                reference_dataset = PredictDataset(self.few_shot_source, transform=self.model.transform)
                dataloader = DataLoader(reference_dataset, batch_size=1, shuffle=False)
            else:
                logger.info("Collecting reference images from training dataset")
                dataloader = self.trainer.datamodule.train_dataloader()

            ref_images = self.collect_reference_images(dataloader)

        # call setup to initialize the model
        self.model.setup(self.class_name, ref_images)

    def _get_class_name(self) -> str:
        """Set the class name used in the prompt ensemble.

        - When a class name is provided by the user, it is used.
        - When the user did not provide a class name, the category name from the datamodule is used, if available.
        - When the user did not provide a class name and the datamodule does not have a category name, the default
            class name "object" is used.
        """
        if self.class_name is not None:
            logger.info("Using class name from init args: %s", self.class_name)
            return self.class_name
        if getattr(self, "_trainer", None) and hasattr(self.trainer.datamodule, "category"):
            logger.info("No class name provided, using category from datamodule: %s", self.trainer.datamodule.category)
            return self.trainer.datamodule.category
        logger.info("No class name provided and no category name found in datamodule using default: object")
        return "object"

    def collect_reference_images(self, dataloader: DataLoader) -> torch.Tensor:
        """Collect reference images for few-shot inference.

        The reference images are collected by iterating the training dataset until the required number of images are
        collected.

        Returns:
            ref_images (Tensor): A tensor containing the reference images.
        """
        ref_images = torch.Tensor()
        for batch in dataloader:
            images = batch["image"][: self.k_shot - ref_images.shape[0]]
            ref_images = torch.cat((ref_images, images))
            if self.k_shot == ref_images.shape[0]:
                break
        return ref_images

    @staticmethod
    def configure_optimizers() -> None:
        """WinCLIP doesn't require optimization, therefore returns no optimizers."""
        return

    def validation_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> dict:
        """Validation Step of WinCLIP."""
        del args, kwargs  # These variables are not used.
        batch["pred_scores"], batch["anomaly_maps"] = self.model(batch["image"])
        return batch

    @property
    def trainer_arguments(self) -> dict[str, int | float]:
        """Set model-specific trainer arguments."""
        return {}

    @property
    def learning_type(self) -> LearningType:
        """The learning type of the model.

        WinCLIP is a zero-/few-shot model, depending on the user configuration. Therefore, the learning type is
        set to ``LearningType.FEW_SHOT`` when ``k_shot`` is greater than zero and ``LearningType.ZERO_SHOT`` otherwise.
        """
        return LearningType.FEW_SHOT if self.k_shot else LearningType.ZERO_SHOT

    def state_dict(self) -> OrderedDict[str, Any]:
        """Return the state dict of the model.

        Before returning the state dict, we remove the parameters of the frozen backbone to reduce the size of the
        checkpoint.
        """
        state_dict = super().state_dict()
        for pattern in self.EXCLUDE_FROM_STATE_DICT:
            remove_keys = [key for key in state_dict if key.startswith(pattern)]
            for key in remove_keys:
                state_dict.pop(key)
        return state_dict

    def load_state_dict(self, state_dict: OrderedDict[str, Any], strict: bool = True) -> Any:  # noqa: ANN401
        """Load the state dict of the model.

        Before loading the state dict, we restore the parameters of the frozen backbone to ensure that the model
        is loaded correctly. We also restore the auxiliary objects like threshold classes and normalization metrics.
        """
        # restore the parameters of the excluded modules, if any
        full_dict = super().state_dict()
        for pattern in self.EXCLUDE_FROM_STATE_DICT:
            restore_dict = {key: value for key, value in full_dict.items() if key.startswith(pattern)}
            state_dict.update(restore_dict)
        return super().load_state_dict(state_dict, strict)

    @staticmethod
    def configure_transforms(image_size: tuple[int, int] | None = None) -> Transform:
        """Configure the default transforms used by the model."""
        if image_size is not None:
            logger.warning("Image size is not used in WinCLIP. The input image size is determined by the model.")
        return Compose(
            [
                Resize((240, 240), antialias=True, interpolation=InterpolationMode.BICUBIC),
                Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ],
        )
