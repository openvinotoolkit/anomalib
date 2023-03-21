"""Implements custom trainer for Anomalib."""


import logging
import warnings
from typing import Optional

from pytorch_lightning import Trainer

from anomalib.models.components.base.anomaly_module import AnomalyModule
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.training.strategies.default.fit import AnomalibFitLoop
from anomalib.training.strategies.default.predict import AnomalibPredictionLoop
from anomalib.training.strategies.default.test import AnomalibTestLoop
from anomalib.training.strategies.default.validate import AnomalibValidationLoop
from anomalib.training.utils import Normalizer, PostProcessor

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
        threshold_method: ThresholdMethod,
        normalization_method: NormalizationMethod,
        manual_image_threshold: Optional[float] = None,
        manual_pixel_threshold: Optional[float] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.lightning_module: AnomalyModule  # for mypy

        self.fit_loop = AnomalibFitLoop(
            min_epochs=kwargs.get("min_epochs", 0), max_epochs=kwargs.get("max_epochs", None)
        )
        self.validate_loop = AnomalibValidationLoop()
        self.test_loop = AnomalibTestLoop()
        self.predict_loop = AnomalibPredictionLoop()

        # TODO: configure these from the config
        self.post_processor = PostProcessor(
            threshold_method=threshold_method,
            manual_image_threshold=manual_image_threshold,
            manual_pixel_threshold=manual_pixel_threshold,
        )
        self.normalizer = Normalizer(normalization_method=normalization_method)

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
        # ------------------------------------------------------------
        self._call_callback_hooks("setup", stage=fn)
        self._call_lightning_module_hook("setup", stage=fn)

        self.strategy.barrier("post_setup")
