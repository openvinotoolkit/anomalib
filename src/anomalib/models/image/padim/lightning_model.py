"""PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization.

Paper https://arxiv.org/abs/2011.08785
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.transforms.v2 import Compose, Normalize, Resize, Transform

from anomalib import LearningType
from anomalib.models.components import AnomalyModule, MemoryBankMixin

from .torch_model import PadimModel

logger = logging.getLogger(__name__)

__all__ = ["Padim"]


class Padim(MemoryBankMixin, AnomalyModule):
    """PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization.

    Args:
        backbone (str): Backbone CNN network
            Defaults to ``resnet18``.
        layers (list[str]): Layers to extract features from the backbone CNN
            Defaults to ``["layer1", "layer2", "layer3"]``.
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
            Defaults to ``True``.
        n_features (int, optional): Number of features to retain in the dimension reduction step.
            Default values from the paper are available for: resnet18 (100), wide_resnet50_2 (550).
            Defaults to ``None``.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: list[str] = ["layer1", "layer2", "layer3"],  # noqa: B006
        pre_trained: bool = True,
        n_features: int | None = None,
    ) -> None:
        super().__init__()

        self.model: PadimModel = PadimModel(
            backbone=backbone,
            pre_trained=pre_trained,
            layers=layers,
            n_features=n_features,
        )

        self.stats: list[torch.Tensor] = []
        self.embeddings: list[torch.Tensor] = []

    @staticmethod
    def configure_optimizers() -> None:
        """PADIM doesn't require optimization, therefore returns no optimizers."""
        return

    def training_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> None:
        """Perform the training step of PADIM. For each batch, hierarchical features are extracted from the CNN.

        Args:
            batch (dict[str, str | torch.Tensor]): Batch containing image filename, image, label and mask
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            Hierarchical feature map
        """
        del args, kwargs  # These variables are not used.

        embedding = self.model(batch["image"])
        self.embeddings.append(embedding.cpu())

    def fit(self) -> None:
        """Fit a Gaussian to the embedding collected from the training set."""
        logger.info("Aggregating the embedding extracted from the training set.")
        embeddings = torch.vstack(self.embeddings)

        logger.info("Fitting a Gaussian to the embedding collected from the training set.")
        self.stats = self.model.gaussian.fit(embeddings)

    def validation_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Perform a validation step of PADIM.

        Similar to the training step, hierarchical features are extracted from the CNN for each batch.

        Args:
            batch (dict[str, str | torch.Tensor]): Input batch
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            Dictionary containing images, features, true labels and masks.
            These are required in `validation_epoch_end` for feature concatenation.
        """
        del args, kwargs  # These variables are not used.

        batch["anomaly_maps"] = self.model(batch["image"])
        return batch

    @property
    def trainer_arguments(self) -> dict[str, int | float]:
        """Return PADIM trainer arguments.

        Since the model does not require training, we limit the max_epochs to 1.
        Since we need to run training epoch before validation, we also set the sanity steps to 0
        """
        return {"max_epochs": 1, "val_check_interval": 1.0, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS

    @staticmethod
    def configure_transforms(image_size: tuple[int, int] | None = None) -> Transform:
        """Default transform for Padim."""
        image_size = image_size or (256, 256)
        return Compose(
            [
                Resize(image_size, antialias=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        )
