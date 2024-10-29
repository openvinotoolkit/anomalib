"""SuperSimpleNet: Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection

Paper https://arxiv.org/pdf/2408.03143
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from anomalib import LearningType
from anomalib.data import Batch

from anomalib.models import AnomalyModule


class SuperSimpleNet(AnomalyModule):
    def __init__(self):
        super().__init__()
        pass

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        pass

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        pass

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        pass

    def configure_optimizers(self) -> OptimizerLRScheduler:
        pass

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS
