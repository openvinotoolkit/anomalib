"""Implements custom trainer for Anomalib."""


import logging
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from lightning_fabric.utilities.types import _PATH
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.loggers import Logger
from pytorch_lightning.plugins import PLUGIN_INPUT
from pytorch_lightning.profilers import Profiler
from pytorch_lightning.strategies import Strategy
from pytorch_lightning.trainer.connectors.accelerator_connector import _LITERAL_WARN, _PRECISION_INPUT

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
    def __init__(
        self,
        logger: Union[Logger, Iterable[Logger], bool] = True,
        enable_checkpointing: bool = True,
        callbacks: Optional[Union[List[Callback], Callback]] = None,
        default_root_dir: Optional[_PATH] = None,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
        num_nodes: int = 1,
        num_processes: Optional[int] = None,
        devices: Optional[Union[List[int], str, int]] = None,
        gpus: Optional[Union[List[int], str, int]] = None,
        auto_select_gpus: Optional[bool] = None,
        tpu_cores: Optional[Union[List[int], str, int]] = None,
        ipus: Optional[int] = None,
        enable_progress_bar: bool = True,
        overfit_batches: Union[int, float] = 0,
        track_grad_norm: Union[int, float, str] = -1,
        check_val_every_n_epoch: Optional[int] = 1,
        fast_dev_run: Union[int, bool] = False,
        accumulate_grad_batches: Optional[Union[int, Dict[int, int]]] = None,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        max_steps: int = -1,
        min_steps: Optional[int] = None,
        max_time: Optional[Union[str, timedelta, Dict[str, int]]] = None,
        limit_train_batches: Optional[Union[int, float]] = None,
        limit_val_batches: Optional[Union[int, float]] = None,
        limit_test_batches: Optional[Union[int, float]] = None,
        limit_predict_batches: Optional[Union[int, float]] = None,
        val_check_interval: Optional[Union[int, float]] = None,
        log_every_n_steps: int = 50,
        accelerator: Optional[Union[str, Accelerator]] = None,
        strategy: Optional[Union[str, Strategy]] = None,
        sync_batchnorm: bool = False,
        precision: _PRECISION_INPUT = 32,
        enable_model_summary: bool = True,
        num_sanity_val_steps: int = 2,
        resume_from_checkpoint: Optional[Union[Path, str]] = None,
        profiler: Optional[Union[Profiler, str]] = None,
        benchmark: Optional[bool] = None,
        deterministic: Optional[Union[bool, _LITERAL_WARN]] = None,
        reload_dataloaders_every_n_epochs: int = 0,
        auto_lr_find: Union[bool, str] = False,
        replace_sampler_ddp: bool = True,
        detect_anomaly: bool = False,
        auto_scale_batch_size: Union[str, bool] = False,
        plugins: Optional[Union[PLUGIN_INPUT, List[PLUGIN_INPUT]]] = None,
        amp_backend: Optional[str] = None,
        amp_level: Optional[str] = None,
        move_metrics_to_cpu: bool = False,
        multiple_trainloader_mode: str = "max_size_cycle",
        inference_mode: bool = True,
    ) -> None:
        super().__init__(
            logger,
            enable_checkpointing,
            callbacks,
            default_root_dir,
            gradient_clip_val,
            gradient_clip_algorithm,
            num_nodes,
            num_processes,
            devices,
            gpus,
            auto_select_gpus,
            tpu_cores,
            ipus,
            enable_progress_bar,
            overfit_batches,
            track_grad_norm,
            check_val_every_n_epoch,
            fast_dev_run,
            accumulate_grad_batches,
            max_epochs,
            min_epochs,
            max_steps,
            min_steps,
            max_time,
            limit_train_batches,
            limit_val_batches,
            limit_test_batches,
            limit_predict_batches,
            val_check_interval,
            log_every_n_steps,
            accelerator,
            strategy,
            sync_batchnorm,
            precision,
            enable_model_summary,
            num_sanity_val_steps,
            resume_from_checkpoint,
            profiler,
            benchmark,
            deterministic,
            reload_dataloaders_every_n_epochs,
            auto_lr_find,
            replace_sampler_ddp,
            detect_anomaly,
            auto_scale_batch_size,
            plugins,
            amp_backend,
            amp_level,
            move_metrics_to_cpu,
            multiple_trainloader_mode,
            inference_mode,
        )

        self.lightning_module: AnomalyModule  # for mypy

        self.fit_loop = AnomalibFitLoop(min_epochs=min_epochs, max_epochs=max_epochs)
        self.validate_loop = AnomalibValidationLoop()
        self.test_loop = AnomalibTestLoop()
        self.predict_loop = AnomalibPredictionLoop()

        # TODO: configure these from the config
        self.post_processor = PostProcessor(threshold_method=ThresholdMethod.ADAPTIVE)
        self.normalizer = Normalizer(normalization_method=NormalizationMethod.MIN_MAX)

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
