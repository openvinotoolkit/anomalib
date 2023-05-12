"""Implements custom trainer for Anomalib."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import logging
import warnings

from pytorch_lightning import Trainer

from anomalib.data import TaskType
from anomalib.models.components.base.anomaly_module import AnomalyModule
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.post_processing.visualizer import VisualizationMode
from anomalib.trainer.loops.one_class import FitLoop, PredictionLoop, TestLoop, ValidationLoop
from anomalib.trainer.utils import (
    CheckpointConnector,
    MetricsManager,
    PostProcessor,
    Thresholder,
    VisualizationStage,
    Visualizer,
    get_normalizer,
)
from anomalib.utils.metrics import AnomalyScoreThreshold

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
        threshold_method (ThresholdMethod): Thresholding method for normalizer.
        normalization_method (NormalizationMethod): Normalization method
        manual_image_threshold (Optional[float]): If threshold method is manual, this needs to be set. Defaults to None.
        manual_pixel_threshold (Optional[float]): If threshold method is manual, this needs to be set. Defaults to None.
        visualization_mode (VisualizationMode): Visualization mode. Options ["full", "simple"]. Defaults to "full".
        show_images (bool): Whether to show images. Defaults to False.
        log_images (bool): Whether to log images. Defaults to False.
        visualization_stage (VisualizationStage): The stage at which to write images to the logger(s).
            Defaults to VisualizationStage.TEST.
    """

    def __init__(
        self,
        threshold_method: ThresholdMethod = ThresholdMethod.ADAPTIVE,
        normalization_method: NormalizationMethod = NormalizationMethod.MIN_MAX,
        manual_image_threshold: float | None = None,
        manual_pixel_threshold: float | None = None,
        image_metrics: list[str] | None = None,
        pixel_metrics: list[str] | None = None,
        visualization_mode: VisualizationMode = VisualizationMode.FULL,
        show_images: bool = False,
        log_images: bool = False,
        visualization_stage: VisualizationStage = VisualizationStage.TEST,
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
        self.image_threshold = AnomalyScoreThreshold().cpu()
        self.pixel_threshold = AnomalyScoreThreshold().cpu()

        self.thresholder = Thresholder(
            trainer=self,
            threshold_method=threshold_method,
            manual_image_threshold=manual_image_threshold,
            manual_pixel_threshold=manual_pixel_threshold,
        )
        self.post_processor = PostProcessor(trainer=self)
        self.normalizer = get_normalizer(trainer=self, normalization_method=normalization_method)
        self.visualizer = Visualizer(
            trainer=self,
            mode=visualization_mode,
            show_images=show_images,
            log_images=log_images,
            stage=visualization_stage,
        )
        self.metrics = MetricsManager(trainer=self, image_metrics=image_metrics, pixel_metrics=pixel_metrics)
