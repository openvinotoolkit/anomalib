"""Implements custom trainer for Anomalib."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Tuple, Union, cast

import lightning as L
import torch
from lightning import seed_everything
from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy
from lightning.fabric.wrappers import _FabricDataLoader, _unwrap_objects
from lightning.pytorch.callbacks.progress import ProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBar
from lightning.pytorch.trainer.states import RunningStage, TrainerFn, TrainerState, TrainerStatus
from lightning.pytorch.utilities.types import STEP_OUTPUT
from rich import get_console
from rich.logging import RichHandler
from rich.table import Table
from torch.utils.data import DataLoader
from tqdm import tqdm

from anomalib.data import TaskType
from anomalib.data.base.datamodule import AnomalibDataModule
from anomalib.models.components.base import AnomalyModule
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.post_processing.visualizer import VisualizationMode
from anomalib.utils.loggers import ProgressBarMetricLogger

from .connectors import CheckpointConnector
from .loops.one_class.evaluation import EvaluationLoop
from .loops.one_class.fit import FitLoop
from .loops.one_class.validation import ValidationLoop

logger = logging.getLogger(__name__)

for name in logging.root.manager.loggerDict:
    for filter_keys in ["lightning", "torch", "anomalib"]:
        if filter_keys in name:
            _logger = logging.getLogger(name)
            _logger.addHandler(RichHandler(rich_tracebacks=True))


class AnomalibTrainer:
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
        loggers: list[Logger] = [],
        task_type: TaskType = TaskType.SEGMENTATION,
        callbacks: list[L.Callback] | None = None,
        ckpt_path: Path | str | None = None,
        project_path: Path | str | None = None,
        seed: None | int = None,
        min_epochs: int | None = None,
        max_epochs: int | None = 1000,
        max_steps: int | None = None,
        grad_accum_steps: int = 1,
        limit_train_batches: Union[int, float] = float("inf"),
        limit_val_batches: Union[int, float] = float("inf"),
        limit_test_batches: int | float = float("inf"),
        limit_predict_batches: int | float = float("inf"),
        validation_frequency: int = 1,
        use_distributed_sampler: bool = True,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: list[int] | str | int = "auto",
        fast_dev_run: int | bool = False,
        enable_progress_bar: bool = True,
    ) -> None:
        """ """
        self.loggers = loggers
        self.callbacks = [] if callbacks is None else callbacks

        self._progress_bar_metrics = ProgressBarMetricLogger()
        if enable_progress_bar:
            self.callbacks.append(RichProgressBar())
            loggers.append(self._progress_bar_metrics)

        self.fabric = L.Fabric(
            accelerator=accelerator,
            devices=devices,
            callbacks=callbacks,
            loggers=loggers,
            strategy=strategy,
        )
        self.global_step = 0
        self.grad_accum_steps: int = grad_accum_steps
        self.current_epoch = 0

        self.seed = seed
        if self.seed is not None:
            seed_everything(self.seed, workers=True)

        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.should_stop = False

        self.limit_train_batches = limit_train_batches
        self.limit_validation_batches = limit_val_batches
        self.limit_test_batches = limit_test_batches
        self.limit_predict_batches = limit_predict_batches
        self.validation_frequency = validation_frequency
        self.use_distributed_sampler = use_distributed_sampler
        self._current_train_return = []
        self._current_validation_return = []
        self._current_test_return = []

        self.project_dir = Path(project_path) if project_path is not None else self._get_project_dir()
        self.ckpt_path = (
            self.project_dir / ckpt_path if ckpt_path is not None else self.project_dir / "weights" / "lightning"
        )
        self.checkpoint_connector = CheckpointConnector(self, self.ckpt_path)
        self.model: AnomalyModule | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.fit_loop = FitLoop(self, min_epochs, max_epochs)
        self.validate_loop = EvaluationLoop(self, TrainerFn.VALIDATING)
        self.test_loop = EvaluationLoop(self, TrainerFn.TESTING)
        self.predict_loop = EvaluationLoop(self, TrainerFn.PREDICTING)
        self.fast_dev_run = fast_dev_run  # TODO. Compatibility
        self.state = TrainerState()
        self.sanity_checking = False  # TODO compatibility for Lightning Trainer. Does not do anything
        self.num_training_batches = float("inf")  # compatibility for Lightning Trainer.
        self._current_eval_dataloader_idx = 0  # compatibility for Lightning Trainer
        # self.num_val_batches = []  # compatibility for Lightning Trainer.
        self.num_validation_batches = []  # Currently only single dataloader is supported
        self.train_dataloader: DataLoader | None = None
        self.test_dataloaders: list[DataLoader] | None = None
        self.validation_dataloaders: list[DataLoader] | None = None
        self.predict_dataloaders: list[DataLoader] | None = None

    def fit(
        self,
        model: AnomalyModule,
        datamodule: AnomalibDataModule | None = None,
    ):
        """The main entrypoint of the trainer, triggering the actual training.

        Args:
            model: the LightningModule to train.
                Can have the same hooks as :attr:`callbacks` (see :meth:`MyCustomTrainer.__init__`).
            train_loader: the training dataloader. Has to be an iterable returning batches.
            val_loader: the validation dataloader. Has to be an iterable returning batches.
                If not specified, no validation will run.
            ckpt_path: Path to previous checkpoints to resume training from.
                If specified, will always look for the latest checkpoint within the given directory.
        """
        self._run(TrainerFn.FITTING, model, datamodule)

    def validate(
        self,
        model: AnomalyModule,
        datamodule: AnomalibDataModule | None = None,
    ) -> list[STEP_OUTPUT] | None:
        """The validation loop running a single validation epoch.

        Args:
            model: the LightningModule to evaluate
            val_loader: The dataloader yielding the validation batches.
        """

        return self._run(TrainerFn.VALIDATING, model=model, datamodule=datamodule)

    def test(
        self,
        model: AnomalyModule,
        datamodule: AnomalibDataModule | None = None,
    ) -> list[STEP_OUTPUT] | None:
        """The test loop running a single test epoch.

        Args:
            model: the LightningModule to evaluate
            test_loader: The dataloader yielding the test batches.
        """
        return self._run(stage=TrainerFn.TESTING, model=model, datamodule=datamodule)

    def predict(
        self,
        model: AnomalyModule,
        datamodule: AnomalibDataModule | None = None,
    ) -> list[STEP_OUTPUT] | None:
        """The predict loop running a single predict epoch.

        Args:
            model: the LightningModule to evaluate
            predict_loader: The dataloader yielding the predict batches.
        """
        return self._run(stage=TrainerFn.PREDICTING, model=model, datamodule=datamodule)

    def _run(
        self, stage: TrainerFn, model: AnomalyModule, datamodule: AnomalibDataModule | None
    ) -> list[STEP_OUTPUT] | None:
        """Wraps setup and teardown code"""
        outputs: list[STEP_OUTPUT] | None = None
        try:
            logger.info("Starting %s run", stage)
            self.state.fn = stage
            if stage == TrainerFn.FITTING:
                self.training = True
            elif stage == TrainerFn.VALIDATING:
                self.validating = True
            elif stage == TrainerFn.TESTING:
                self.testing = True
            elif stage == TrainerFn.PREDICTING:
                self.predicting = True

            self.fabric.launch()
            self._setup_dataloaders(model, datamodule)
            self._setup(model)
            self.state.status = TrainerStatus.RUNNING

            self.checkpoint_connector.restore()

            if stage == TrainerFn.FITTING:
                self.fabric.call("on_fit_start", trainer=self, pl_module=_unwrap_objects(self.model))
            outputs = getattr(self, f"{stage}_loop").run()
        except Exception as exception:
            logger.exception(exception)
            self.state.status = TrainerStatus.INTERRUPTED
        else:
            self.state.status = TrainerStatus.FINISHED
            self.state.stage = None

            if stage == TrainerFn.FITTING:
                self.fabric.call("on_fit_end", trainer=self, pl_module=_unwrap_objects(self.model))
                self.should_stop = False
                assert self.state.stopped
        finally:
            if stage == TrainerFn.FITTING:
                self.training = False
            elif stage == TrainerFn.VALIDATING:
                self.validating = False
            elif stage == TrainerFn.TESTING:
                self.testing = False
            elif stage == TrainerFn.PREDICTING:
                self.predicting = False
            else:
                raise ValueError(f"Unknown stage {stage}")
        if outputs is not None:
            return outputs

    def _get_project_dir(self) -> Path:
        return Path("results") / "custom_trainer" / "padim" / "runs"

    def progbar_wrapper(self, iterable: Iterable, total: int, **kwargs: Any):
        """Wraps the iterable with tqdm for global rank zero.

        Args:
            iterable: the iterable to wrap with tqdm
            total: the total length of the iterable, necessary in case the number of batches was limited.
        """
        if self.fabric.is_global_zero:
            return tqdm(iterable, total=total, **kwargs)
        return iterable

    def _parse_optimizers_schedulers(
        self, configure_optim_output
    ) -> Tuple[
        Optional[L.fabric.utilities.types.Optimizable],
        Optional[Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]],
    ]:
        """Recursively parses the output of :meth:`lightning.pytorch.LightningModule.configure_optimizers`.

        Args:
            configure_optim_output: The output of ``configure_optimizers``.
                For supported values, please refer to :meth:`lightning.pytorch.LightningModule.configure_optimizers`.
        """
        _lr_sched_defaults = {"interval": "epoch", "frequency": 1, "monitor": "val_loss"}

        # single optimizer
        if isinstance(configure_optim_output, L.fabric.utilities.types.Optimizable):
            return configure_optim_output, None

        # single lr scheduler
        if isinstance(configure_optim_output, L.fabric.utilities.types.LRScheduler):
            return None, _lr_sched_defaults.update(scheduler=configure_optim_output)

        # single lr scheduler config
        if isinstance(configure_optim_output, Mapping):
            _lr_sched_defaults.update(configure_optim_output)
            return None, _lr_sched_defaults

        # list or tuple
        if isinstance(configure_optim_output, (list, tuple)):
            if all(isinstance(_opt_cand, L.fabric.utilities.types.Optimizable) for _opt_cand in configure_optim_output):
                # single optimizer in list
                if len(configure_optim_output) == 1:
                    return configure_optim_output[0][0], None

                raise NotImplementedError("BYOT only supports a single optimizer")

            if all(
                isinstance(_lr_cand, (L.fabric.utilities.types.LRScheduler, Mapping))
                for _lr_cand in configure_optim_output
            ):
                # single scheduler in list
                if len(configure_optim_output) == 1:
                    return None, self._parse_optimizers_schedulers(configure_optim_output[0])[1]

            # optimizer and lr scheduler
            elif len(configure_optim_output) == 2:
                opt_cands, lr_cands = (
                    self._parse_optimizers_schedulers(configure_optim_output[0])[0],
                    self._parse_optimizers_schedulers(configure_optim_output[1])[1],
                )
                return opt_cands, lr_cands

        return None, None

    def print_metrics(self, metrics: dict, prefix: str):
        """Logs the metrics of the anomaly module.

        Args:
            anomaly_module: The anomaly module.
            prefix: The prefix to use for the metric names.
        """
        if self.fabric.is_global_zero:
            table = Table(title=f"{prefix} Metrics", style="cyan")
            table.add_column("Metric", justify="right", style="cyan")
            table.add_column("Value", justify="left", style="green")
            for k, v in metrics.items():
                table.add_row(k, str(v))
            get_console().print(table)

    def save_checkpoint(self, filepath, weights_only: bool = False):
        """Saves the model and trainer state to a checkpoint file.

        The signature is kept same as the Lightning Trainer as `ModelCheckpointCallback` calls this method.

        Args:
            filepath: Write-target.
            weights_only: If True, then only the model weights will be saved.
        """
        # TODO
        if self.fabric.is_global_zero:
            self.checkpoint_connector.save()

    def _setup(self, model: AnomalyModule):
        """Setup model and optimizer using fabric.

        Args:
            model: the AnomalyModule to train.
        """
        if isinstance(self.fabric.strategy, L.fabric.strategies.fsdp.FSDPStrategy):
            # currently, there is no way to support fsdp with model.configure_optimizers in fabric
            # as it would require fabric to hold a reference to the model, which we don't want to.
            raise NotImplementedError("BYOT currently does not support FSDP")

        optimizer, scheduler_cfg = self._parse_optimizers_schedulers(model.configure_optimizers())
        if optimizer is None:
            self.model = self.fabric.setup(model)
        else:
            self.model, self.optimizer = self.fabric.setup(model, optimizer)

        self.scheduler_cfg = scheduler_cfg

    def _setup_dataloaders(self, model: AnomalyModule, datamodule: AnomalibDataModule | None) -> None:
        """Setup dataloaders using fabric.

        Args:
            model (AnomalyModule): AnomalyModule
            datamodule (AnomalibDataModule): Datamodule

        Returns:
            None
        """
        if datamodule is not None:
            datamodule.prepare_data()
            # TODO add a barrier that checks if all processes have finished preparing data
            datamodule.setup(stage=None)  # stage is unused in AnomalibDataModule

        if self.state.stage == RunningStage.TRAINING:
            train_loader = model.train_dataloader() if datamodule is None else datamodule.train_dataloader()
            self.train_dataloader = self.fabric.setup_dataloaders(
                train_loader, use_distributed_sampler=self.use_distributed_sampler
            )
        elif self.state.stage == RunningStage.TESTING:
            test_loader = model.test_dataloader() if datamodule is None else datamodule.test_dataloader()
            self.test_dataloaders = [
                self.fabric.setup_dataloaders(test_loader, use_distributed_sampler=self.use_distributed_sampler)
            ]
        elif self.state.stage == RunningStage.PREDICTING:
            predict_loader = model.predict_dataloader() if datamodule is None else datamodule.predict_dataloader()
            self.predict_dataloaders = [
                self.fabric.setup_dataloaders(predict_loader, use_distributed_sampler=self.use_distributed_sampler)
            ]

        if self.state.stage in (RunningStage.VALIDATING, RunningStage.TRAINING):
            val_loader = model.val_dataloader() if datamodule is None else datamodule.val_dataloader()
            self.validation_dataloaders = [
                self.fabric.setup_dataloaders(val_loader, use_distributed_sampler=self.use_distributed_sampler)
            ]
        # self.predict_dataloaders = [self.fabric.setup_dataloaders(
        #         datamodule, use_distributed_sampler=self.use_distributed_sampler
        #     )]

    def step_scheduler(
        self,
        model: L.LightningModule,
        scheduler_cfg: Optional[Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]],
        level: Literal["step", "epoch"],
        current_value: int,
    ) -> None:
        """Steps the learning rate scheduler if necessary.

        Args:
            model: The LightningModule to train
            scheduler_cfg: The learning rate scheduler configuration.
                Have a look at :meth:`lightning.pytorch.LightninModule.configure_optimizers` for supported values.
            level: whether we are trying to step on epoch- or step-level
            current_value: Holds the current_epoch if ``level==epoch``, else holds the ``global_step``
        """

        # no scheduler
        if scheduler_cfg is None:
            return

        # wrong interval (step vs. epoch)
        if scheduler_cfg["interval"] != level:
            return

        # right interval, but wrong step wrt frequency
        if current_value % cast(int, scheduler_cfg["frequency"]) != 0:
            return

        # assemble potential monitored values
        possible_monitor_vals = {None: None}
        if isinstance(self._current_train_return, torch.Tensor):
            possible_monitor_vals.update("train_loss", self._current_train_return)
        elif isinstance(self._current_train_return, Mapping):
            possible_monitor_vals.update({"train_" + k: v for k, v in self._current_train_return.items()})

        # TODO change this to handle return type of validation_step
        if isinstance(self._current_validation_return, torch.Tensor):
            possible_monitor_vals.update("val_loss", self._current_validation_return)
        elif isinstance(self._current_validation_return, Mapping):
            possible_monitor_vals.update({"val_" + k: v for k, v in self._current_validation_return.items()})

        try:
            monitor = possible_monitor_vals[cast(Optional[str], scheduler_cfg["monitor"])]
        except KeyError as ex:
            possible_keys = list(possible_monitor_vals.keys())
            raise KeyError(
                f"monitor {scheduler_cfg['monitor']} is invalid. Possible values are {possible_keys}."
            ) from ex

        # rely on model hook for actual step
        model.lr_scheduler_step(scheduler_cfg["scheduler"], monitor)

    @property
    def interrupted(self) -> bool:
        return self.state.status == TrainerStatus.INTERRUPTED

    @property
    def training(self) -> bool:
        return self.state.stage == RunningStage.TRAINING

    @training.setter
    def training(self, val: bool) -> None:
        if val:
            self.state.stage = RunningStage.TRAINING
        elif self.training:
            self.state.stage = None

    @property
    def validating(self) -> bool:
        return self.state.stage == RunningStage.VALIDATING

    @validating.setter
    def validating(self, val: bool) -> None:
        if val:
            self.state.stage = RunningStage.VALIDATING
        elif self.validating:
            self.state.stage = None

    @property
    def testing(self) -> bool:
        return self.state.stage == RunningStage.TESTING

    @testing.setter
    def testing(self, val: bool) -> None:
        if val:
            self.state.stage = RunningStage.TESTING
        elif self.testing:
            self.state.stage = None

    @property
    def predicting(self) -> bool:
        return self.state.stage == RunningStage.PREDICTING

    @predicting.setter
    def predicting(self, val: bool) -> None:
        if val:
            self.state.stage = RunningStage.PREDICTING
        elif self.predicting:
            self.state.stage = None

    @property
    def progress_bar_callback(self) -> Optional[ProgressBar]:
        for c in self.callbacks:
            if isinstance(c, ProgressBar):
                return c
        return None

    @property
    def progress_bar_metrics(self) -> dict[str, Any]:
        return self._progress_bar_metrics._metrics

    @property
    def should_validate(self) -> bool:
        """Whether to currently run validation."""
        return self.current_epoch % self.validation_frequency == 0

    @property
    def strategy(self):
        return self.fabric.strategy

    @property
    def is_global_zero(self):
        return self.fabric.is_global_zero

    @property
    def num_val_batches(self):
        """For compatibility with Lightning Trainer."""
        return self.num_validation_batches
