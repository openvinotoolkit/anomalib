"""Towards Total Recall in Industrial Anomaly Detection.

Paper https://arxiv.org/abs/2106.08265.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig, ListConfig
from torch import Tensor

from anomalib.models.components import AnomalyModule, KCenterGreedy, MemoryBankMixin
from anomalib.models.patchcore.torch_model import PatchcoreModel

logger = logging.getLogger(__name__)


class Patchcore(AnomalyModule, MemoryBankMixin):
    """PatchcoreLightning Module to train PatchCore algorithm.

    Args:
        input_size (tuple[int, int]): Size of the model input.
        backbone (str): Backbone CNN network
        layers (list[str]): Layers to extract features from the backbone CNN
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
        coreset_sampling_ratio (float, optional): Coreset sampling ratio to subsample embedding.
            Defaults to 0.1.
        num_neighbors (int, optional): Number of nearest neighbors. Defaults to 9.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        backbone: str = "wide_resnet50_2",
        layers: Sequence[str] = ("layer2", "layer3"),
        pre_trained: bool = True,
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
    ) -> None:
        super().__init__()

        self.model: PatchcoreModel = PatchcoreModel(
            input_size=input_size,
            backbone=backbone,
            pre_trained=pre_trained,
            layers=layers,
            num_neighbors=num_neighbors,
        )
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.embeddings: list[Tensor] = []

    def configure_optimizers(self) -> None:
        """Configure optimizers.

        Returns:
            None: Do not set optimizers by returning None.
        """
        return

    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> None:
        """Generate feature embedding of the batch.

        Args:
            batch (dict[str, str | Tensor]): Batch containing image filename, image, label and mask
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            dict[str, np.ndarray]: Embedding Vector
        """
        del args, kwargs  # These variables are not used.

        # Model is in eval mode to avoid updating the weights.
        self.model.feature_extractor.eval()

        # Generate embedding
        embedding = self.model(batch["image"])
        self.embeddings.append(embedding)

    def fit(self) -> None:
        """Fit the PatchCore model via coreset sampling.

        Args:
            embedding (Tensor | list[Tensor]): Embedding tensor from the CNN
            sampling_ratio (float): Coreset sampling ratio
        """
        embedding = torch.vstack(self.embeddings)

        # Coreset Subsampling
        logger.info("Fitting the PatchCore model via core-set sampling.")
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=self.coreset_sampling_ratio)
        coreset = sampler.sample_coreset()
        self.model.memory_bank = coreset

        # Model is now fitted.
        self._is_fitted = torch.tensor([True])

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Get batch of anomaly maps from input image batch.

        Args:
            batch (dict[str, str | Tensor]): Batch containing image filename, image, label and mask
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: Image filenames, test images, GT and predicted label/masks
        """
        # These variables are not used.
        del args, kwargs

        # Get anomaly maps and predicted scores from the model.
        output = self.model(batch["image"])

        # Add anomaly maps and predicted scores to the batch.
        batch["anomaly_maps"] = output["anomaly_map"]
        batch["pred_scores"] = output["pred_score"]

        return batch

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return Patchcore trainer arguments."""
        return {"gradient_clip_val": 0, "max_epochs": 1, "num_sanity_val_steps": 0}


class PatchcoreLightning(Patchcore):
    """PatchcoreLightning Module to train PatchCore algorithm.

    Args:
        hparams (DictConfig | ListConfig): Model params
    """

    def __init__(self, hparams: DictConfig | ListConfig) -> None:
        super().__init__(
            input_size=hparams.model.input_size,
            backbone=hparams.model.backbone,
            layers=hparams.model.layers,
            pre_trained=hparams.model.pre_trained,
            coreset_sampling_ratio=hparams.model.coreset_sampling_ratio,
            num_neighbors=hparams.model.num_neighbors,
        )
        self.hparams: DictConfig | ListConfig
        self.save_hyperparameters(hparams)
