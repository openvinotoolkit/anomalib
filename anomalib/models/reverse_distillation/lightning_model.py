"""Anomaly Detection via Reverse Distillation from One-Class Embedding.

https://arxiv.org/abs/2201.10703v2
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Tuple

from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from torch import Tensor

from anomalib.models.components import AnomalyModule
from anomalib.models.components.feature_extraction import FeatureExtractorParams

from .loss import ReverseDistillationLoss
from .torch_model import ReverseDistillationModel


@MODEL_REGISTRY
class ReverseDistillation(AnomalyModule):
    """PL Lightning Module for Reverse Distillation Algorithm.

    Args:
        input_size (Tuple[int, int]): Size of model input
        feature_extractor (FeatureExtractorParams): Feature extractor params
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        feature_extractor: FeatureExtractorParams,
        anomaly_map_mode: str,
    ):
        super().__init__()
        self.model = ReverseDistillationModel(
            feature_extractor_params=feature_extractor,
            input_size=input_size,
            anomaly_map_mode=anomaly_map_mode,
        )
        self.loss = ReverseDistillationLoss()

    def training_step(self, batch, _) -> Dict[str, Tensor]:  # type: ignore
        """Training Step of Reverse Distillation Model.

        Features are extracted from three layers of the Encoder model. These are passed to the bottleneck layer
        that are passed to the decoder network. The loss is then calculated based on the cosine similarity between the
        encoder and decoder features.

        Args:
          batch (Tensor): Input batch
          _: Index of the batch.

        Returns:
          Feature Map
        """
        loss = self.loss(*self.model(batch["image"]))
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, _):  # pylint: disable=arguments-differ
        """Validation Step of Reverse Distillation Model.

        Similar to the training step, encoder/decoder features are extracted from the CNN for each batch, and
        anomaly map is computed.

        Args:
          batch (Tensor): Input batch
          _: Index of the batch.

        Returns:
          Dictionary containing images, anomaly maps, true labels and masks.
          These are required in `validation_epoch_end` for feature concatenation.
        """
        batch["anomaly_maps"] = self.model(batch["image"])
        return batch
