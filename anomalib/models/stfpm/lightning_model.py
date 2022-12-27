"""STFPM: Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection.

https://arxiv.org/abs/2103.04257
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

from pytorch_lightning.utilities.cli import MODEL_REGISTRY

from anomalib.models.components import AnomalyModule
from anomalib.models.components.feature_extraction import FeatureExtractorParams
from anomalib.models.stfpm.loss import STFPMLoss
from anomalib.models.stfpm.torch_model import STFPMModel


@MODEL_REGISTRY
class Stfpm(AnomalyModule):
    """PL Lightning Module for the STFPM algorithm.

    Args:
        input_size (Tuple[int, int]): Size of the model input.
        student_teacher_model (FeatureExtractorParams): Parameters for teacher and student models.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        student_teacher_model: FeatureExtractorParams,
    ):
        super().__init__()

        self.model = STFPMModel(
            input_size=input_size,
            student_teacher_model_params=student_teacher_model,
        )
        self.loss = STFPMLoss()

    def training_step(self, batch, _):  # pylint: disable=arguments-differ
        """Training Step of STFPM.

        For each batch, teacher and student and teacher features are extracted from the CNN.

        Args:
          batch (Tensor): Input batch
          _: Index of the batch.

        Returns:
          Loss value
        """
        self.model.teacher_model.eval()
        teacher_features, student_features = self.model.forward(batch["image"])
        loss = self.loss(teacher_features, student_features)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, _):  # pylint: disable=arguments-differ
        """Validation Step of STFPM.

        Similar to the training step, student/teacher features are extracted from the CNN for each batch, and
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
