"""Implements custom trainer for Anomalib."""

from __future__ import annotations

import logging
import warnings
from typing import Optional

from pytorch_lightning import Trainer

from anomalib.data import TaskType
from anomalib.models.components.base.anomaly_module import AnomalyModule
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.training.learning_strategies.default.fit import AnomalibFitLoop
from anomalib.training.learning_strategies.default.predict import AnomalibPredictionLoop
from anomalib.training.learning_strategies.default.test import AnomalibTestLoop
from anomalib.training.learning_strategies.default.validate import AnomalibValidationLoop
from anomalib.training.utils import MetricsManager, Normalizer, PostProcessor

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
    """

    def __init__(
        self,
        threshold_method: ThresholdMethod = ThresholdMethod.ADAPTIVE,
        normalization_method: NormalizationMethod = NormalizationMethod.MIN_MAX,
        manual_image_threshold: Optional[float] = None,
        manual_pixel_threshold: Optional[float] = None,
        image_metrics: list[str] | None = None,
        pixel_metrics: list[str] | None = None,
        task_type: TaskType = TaskType.SEGMENTATION,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.lightning_module: AnomalyModule  # for mypy

        self.fit_loop = AnomalibFitLoop(
            min_epochs=kwargs.get("min_epochs", 0), max_epochs=kwargs.get("max_epochs", None)
        )
        self.validate_loop = AnomalibValidationLoop()
        self.test_loop = AnomalibTestLoop()
        self.predict_loop = AnomalibPredictionLoop()

        self.task_type = task_type

        # TODO: configure these from the config
        self.post_processor = PostProcessor(
            threshold_method=threshold_method,
            manual_image_threshold=manual_image_threshold,
            manual_pixel_threshold=manual_pixel_threshold,
        )
        self.normalizer = Normalizer(normalization_method=normalization_method)
        self.metrics_manager = MetricsManager(image_metrics=image_metrics, pixel_metrics=pixel_metrics)

    def _call_setup_hook(self) -> None:
        """Override the setup hook to call setup for required anomalib classes.

        Ensures that the necessary attributes are added to the model before callbacks are called. The majority of the
        code is same as the base class. Add custom setup only between the commented block.
        """
        assert self.state.fn is not None
        fn = self.state.fn

        self.strategy.barrier("pre_setup")

        if self.datamodule is not None:
            self._call_lightning_datamodule_hook("setup", stage=fn)

        # Setup required classes before callbacks and lightning module
        self.post_processor.setup(self.lightning_module)
        self.normalizer.setup()
        self.metrics_manager.setup(self.lightning_module, self.task_type)
        # ------------------------------------------------------------
        self._call_callback_hooks("setup", stage=fn)
        self._call_lightning_module_hook("setup", stage=fn)

        self.strategy.barrier("post_setup")
