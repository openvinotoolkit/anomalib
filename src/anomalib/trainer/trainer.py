"""Implements custom trainer for Anomalib."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging
import warnings

from pytorch_lightning import Trainer

from anomalib.data import TaskType
from anomalib.models.components.base.anomaly_module import AnomalyModule
from anomalib.post_processing import NormalizationMethod
from anomalib.trainer.loops.one_class import FitLoop, PredictionLoop, TestLoop, ValidationLoop
from anomalib.trainer.utils import CheckpointConnector, MetricsManager, PostProcessor, Thresholder, get_normalizer
from anomalib.utils.metrics import BaseAnomalyThreshold

log = logging.getLogger(__name__)
# warnings to ignore in trainer
warnings.filterwarnings(
    "ignore", message="torch.distributed.reduce_op is deprecated, please use torch.distributed.ReduceOp instead"
)


class AnomalibTrainer(Trainer):
    """Anomalib trainer.

    Note:
        Refer to PyTorch Lightning's Trainer for a list of parameters for details on other Trainer parameters.

    Args:
        image_threshold_method (dict): Thresholding method. If None, adaptive thresholding is used.
        pixel_threshold_method (dict): Thresholding method. If None, adaptive thresholding is used.
        normalization_method (NormalizationMethod): Normalization method
    """

    def __init__(
        self,
        image_threshold: dict | None = None,  # TODO change from dict to BaseAnomalyThreshold in CLI
        pixel_threshold: dict | None = None,
        normalization_method: NormalizationMethod = NormalizationMethod.MIN_MAX,
        image_metrics: list[str] | None = None,
        pixel_metrics: list[str] | None = None,
        task_type: TaskType = TaskType.SEGMENTATION,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._checkpoint_connector = CheckpointConnector(self, kwargs.get("resume_from_checkpoint", None))

        self.lightning_module: AnomalyModule  # for mypy

        self.fit_loop = FitLoop(min_epochs=kwargs.get("min_epochs", 0), max_epochs=kwargs.get("max_epochs", None))
        self.validate_loop = ValidationLoop()
        self.test_loop = TestLoop()
        self.predict_loop = PredictionLoop()

        self.task_type = task_type
        # these are part of the trainer as they are used in the metrics-manager, post-processor and thresholder
        self.image_threshold = BaseAnomalyThreshold()
        self.pixel_threshold = BaseAnomalyThreshold()

        self.thresholder = Thresholder(
            trainer=self,
            image_threshold_method=image_threshold,
            pixel_threshold_method=pixel_threshold,
        )
        self.post_processor = PostProcessor(trainer=self)
        self.normalizer = get_normalizer(trainer=self, normalization_method=normalization_method)
        self.metrics = MetricsManager(trainer=self, image_metrics=image_metrics, pixel_metrics=pixel_metrics)
