"""Implements custom trainer for Anomalib."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Tuple, Union, cast

import lightning as L
import torch
from lightning import seed_everything
from lightning.fabric.loggers import Logger
from lightning.fabric.wrappers import _unwrap_objects
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning_utilities import apply_to_collection
from tqdm import tqdm

from anomalib.data import TaskType
from anomalib.data.base.datamodule import AnomalibDataModule
from anomalib.models.components.base import AnomalyModule
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.post_processing.visualizer import VisualizationMode

from .connectors import CheckpointConnector


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
        loggers: list[Logger] | None = [],
        task_type: TaskType = TaskType.SEGMENTATION,
        callbacks: list[L.Callback] | None = None,
        ckpt_path: Path | str | None = None,
        project_path: Path | str | None = None,
        seed: None | int = None,
        max_epochs: Optional[int] = 1000,
        max_steps: Optional[int] = None,
        grad_accum_steps: int = 1,
        limit_train_batches: Union[int, float] = float("inf"),
        limit_val_batches: Union[int, float] = float("inf"),
        limit_test_batches: int | float = float("inf"),
        validation_frequency: int = 1,
        use_distributed_sampler: bool = True,
    ) -> None:
        """ """
        self.loggers = loggers
        self.fabric = L.Fabric(
            accelerator="gpu",
            devices=1,
            callbacks=callbacks,
            loggers=loggers,
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

        # ensures limit_X_batches is either int or inf
        if not isinstance(limit_train_batches, int):
            assert limit_train_batches == float("inf")

        if not isinstance(limit_val_batches, int):
            assert limit_val_batches == float("inf")

        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.limit_test_batches = limit_test_batches
        self.validation_frequency = validation_frequency
        self.use_distributed_sampler = use_distributed_sampler
        self._current_train_return = []
        self._current_val_return = []
        self._current_test_return = []

        self.project_dir = Path(project_path) if project_path is not None else self._get_project_dir()
        self.ckpt_path = (
            self.project_dir / ckpt_path if ckpt_path is not None else self.project_dir / "weights" / "lightning"
        )
        self.checkpoint_connector = CheckpointConnector(self, self.ckpt_path)

    def _get_project_dir(self) -> Path:
        return Path("results") / "custom_trainer" / "padim" / "runs"

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
        self.fabric.launch()

        # setup dataloaders
        if datamodule is not None:
            datamodule.prepare_data()
            # TODO add a barrier that checks if all processes have finished preparing data
            datamodule.setup("fit")
        train_loader = model.train_dataloader() if datamodule is None else datamodule.train_dataloader()
        val_loader = model.val_dataloader() if datamodule is None else datamodule.val_dataloader()
        train_loader = self.fabric.setup_dataloaders(train_loader, use_distributed_sampler=self.use_distributed_sampler)
        if val_loader is not None:
            val_loader = self.fabric.setup_dataloaders(val_loader, use_distributed_sampler=self.use_distributed_sampler)

        # setup model and optimizer
        if isinstance(self.fabric.strategy, L.fabric.strategies.fsdp.FSDPStrategy):
            # currently, there is no way to support fsdp with model.configure_optimizers in fabric
            # as it would require fabric to hold a reference to the model, which we don't want to.
            raise NotImplementedError("BYOT currently does not support FSDP")

        optimizer, scheduler_cfg = self._parse_optimizers_schedulers(model.configure_optimizers())
        if optimizer is None:
            model = self.fabric.setup(model)
        else:
            model, optimizer = self.fabric.setup(model, optimizer)

        # assemble state (current epoch and global step will be added in save)
        state = {"model": model, "optim": optimizer, "scheduler": scheduler_cfg}  # todo move to dump checkpoint

        # load best checkpoint if available
        self.checkpoint_connector.restore(state, self.ckpt_path)

        # check if we even need to train here
        if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
            self.should_stop = True

        # setup train callback
        self.fabric.call("setup", trainer=self, pl_module=_unwrap_objects(model), stage="fit")

        while not self.should_stop:
            self.train_loop(
                model, optimizer, train_loader, limit_batches=self.limit_train_batches, scheduler_cfg=scheduler_cfg
            )

            if self.should_validate:
                self.val_loop(model, val_loader, limit_batches=self.limit_val_batches)

            self.step_scheduler(model, scheduler_cfg, level="epoch", current_value=self.current_epoch)

            self.current_epoch += 1

            # stopping condition on epoch level
            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.should_stop = True

            self.checkpoint_connector.save(state)

        # reset for next fit call
        if datamodule is not None:
            datamodule.teardown("fit")
        self.fabric.call("teardown", trainer=self, pl_module=_unwrap_objects(model), stage="fit")
        self.should_stop = False

    def train_loop(
        self,
        model: L.LightningModule,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        limit_batches: Union[int, float] = float("inf"),
        scheduler_cfg: Optional[Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]] = None,
    ):
        """The training loop running a single training epoch.

        Args:
            model: the LightningModule to train
            optimizer: the optimizer, optimizing the LightningModule.
            train_loader: The dataloader yielding the training batches.
            limit_batches: Limits the batches during this training epoch.
                If greater then the number of batches in the ``train_loader``, this has no effect.
            scheduler_cfg: The learning rate scheduler configuration.
                Have a look at :meth:`lightning.pytorch.LightninModule.configure_optimizers` for supported values.
        """
        self.fabric.call("on_train_epoch_start", trainer=self, pl_module=_unwrap_objects(model))
        iterable = self.progbar_wrapper(
            train_loader, total=min(len(train_loader), limit_batches), desc=f"Epoch {self.current_epoch}"
        )
        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                self.fabric.call("on_train_epoch_end", trainer=self.fabric, pl_module=_unwrap_objects(model))
                return

            self.fabric.call(
                "on_train_batch_start",
                batch=batch,
                batch_idx=batch_idx,
                trainer=self.fabric,
                pl_module=_unwrap_objects(model),
            )

            # check if optimizer should step in gradient accumulation
            should_optim_step = self.global_step % self.grad_accum_steps == 0 and optimizer is not None
            if should_optim_step:
                # currently only supports a single optimizer
                self.fabric.call("on_before_optimizer_step", optimizer, 0)

                # optimizer step runs train step internally through closure
                optimizer.step(partial(self.training_step, model=model, batch=batch, batch_idx=batch_idx))
                self.fabric.call("on_before_zero_grad", optimizer)

                optimizer.zero_grad()

            else:
                # gradient accumulation -> no optimizer step
                self.training_step(model=model, batch=batch, batch_idx=batch_idx)

            self.fabric.call(
                "on_train_batch_end",
                outputs=self._current_train_return,
                batch=batch,
                batch_idx=batch_idx,
                trainer=self,
                pl_module=_unwrap_objects(model),
            )

            # this guard ensures, we only step the scheduler once per global step
            if should_optim_step:
                self.step_scheduler(model, scheduler_cfg, level="step", current_value=self.global_step)

            # add output values to progress bar
            self._format_iterable(iterable, self._current_train_return, "train")

            # only increase global step if optimizer stepped
            self.global_step += int(should_optim_step)

            # stopping criterion on step level
            if self.max_steps is not None and self.global_step >= self.max_steps:
                self.should_stop = True
                break

        self.fabric.call("on_train_epoch_end", trainer=self.fabric, pl_module=_unwrap_objects(model))

    def val_loop(
        self,
        model: AnomalyModule,
        val_loader: Optional[torch.utils.data.DataLoader],
        limit_batches: Union[int, float] = float("inf"),
    ):
        """The validation loop ruunning a single validation epoch.

        Args:
            model: the LightningModule to evaluate
            val_loader: The dataloader yielding the validation batches.
            limit_batches: Limits the batches during this validation epoch.
                If greater then the number of batches in the ``val_loader``, this has no effect.
        """
        # no validation if val_loader wasn't passed
        if val_loader is None:
            return

        # no validation but warning if val_loader was passed, but validation_step not implemented
        if val_loader is not None and not is_overridden("validation_step", _unwrap_objects(model), AnomalyModule):
            L.fabric.utilities.rank_zero_warn(
                "Your LightningModule does not have a validation_step implemented, "
                "but you passed a validation dataloder. Skipping Validation."
            )
            return

        self.fabric.call("on_validation_model_eval")  # calls `model.eval()`

        # setup callbacks
        self.fabric.call("setup", trainer=self, pl_module=_unwrap_objects(model), stage="validate")

        torch.set_grad_enabled(False)

        self.fabric.call("on_validation_epoch_start", trainer=self, pl_module=_unwrap_objects(model))

        iterable = self.progbar_wrapper(val_loader, total=min(len(val_loader), limit_batches), desc="Validation")

        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            self.fabric.call(
                "on_validation_batch_start",
                batch=batch,
                batch_idx=batch_idx,
                dataloader_idx=0,
                trainer=self.fabric,
                pl_module=_unwrap_objects(model),
            )

            out = model.validation_step(batch, batch_idx)
            # avoid gradients in stored/accumulated values -> prevents potential OOM
            out = apply_to_collection(out, torch.Tensor, lambda x: x.detach())

            out = model.validation_step_end(out)  # TODO change this

            self.fabric.call(
                "on_validation_batch_end",
                outputs=out,
                batch=batch,
                batch_idx=batch_idx,
                dataloader_idx=0,
                trainer=self,
                pl_module=_unwrap_objects(model),
            )
            self._current_val_return.append(out)

            # self._format_iterable(iterable, self._current_val_return, "val")
            # TODO compute metrics here

        model.validation_epoch_end(self._current_val_return)  # TODO change this
        self.print_metrics(iterable, _unwrap_objects(model), "val")
        self.fabric.call("on_validation_epoch_end", trainer=self, pl_module=_unwrap_objects(model))

        self.fabric.call("on_validation_model_train")
        # teardown callbacks
        self.fabric.call("teardown", trainer=self, pl_module=_unwrap_objects(model), stage="validate")
        torch.set_grad_enabled(True)

    def training_step(self, model: L.LightningModule, batch: Any, batch_idx: int) -> torch.Tensor | None:
        """A single training step, running forward and backward. The optimizer step is called separately, as this
        is given as a closure to the optimizer step.

        Args:
            model: the lightning module to train
            batch: the batch to run the forward on
            batch_idx: index of the current batch w.r.t the current epoch
        """
        outputs: Union[torch.Tensor, Mapping[str, Any]] = model.training_step(batch, batch_idx=batch_idx)
        if outputs is None:
            return None

        loss = outputs if isinstance(outputs, torch.Tensor) else outputs["loss"]

        self.fabric.call("on_before_backward", loss)
        self.fabric.backward(loss)
        self.fabric.call("on_after_backward")

        # avoid gradients in stored/accumulated values -> prevents potential OOM
        self._current_train_return = apply_to_collection(outputs, dtype=torch.Tensor, function=lambda x: x.detach())

        return loss

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
        if isinstance(self._current_val_return, torch.Tensor):
            possible_monitor_vals.update("val_loss", self._current_val_return)
        elif isinstance(self._current_val_return, Mapping):
            possible_monitor_vals.update({"val_" + k: v for k, v in self._current_val_return.items()})

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
    def should_validate(self) -> bool:
        """Whether to currently run validation."""
        return self.current_epoch % self.validation_frequency == 0

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

    def print_metrics(self, iterable, anomaly_module: AnomalyModule, prefix: str):
        """Logs the metrics of the anomaly module.

        Args:
            anomaly_module: The anomaly module.
            prefix: The prefix to use for the metric names.
        """
        if isinstance(iterable, tqdm):
            tqdm.write(str({key: value.compute() for key, value in anomaly_module.pixel_metrics.items()}))
        if self.fabric.is_global_zero:
            if anomaly_module.pixel_metrics.update_called:
                self._format_iterable(
                    iterable,
                    {key: value.compute() for key, value in anomaly_module.pixel_metrics.items()},
                    prefix=f"{prefix}_pixel",
                )
            else:
                self._format_iterable(
                    iterable,
                    {key: value.compute() for key, value in anomaly_module.image_metrics.items()},
                    prefix=f"{prefix}_image",
                )

    @staticmethod
    def _format_iterable(
        prog_bar, candidates: Optional[Union[torch.Tensor, Mapping[str, Union[torch.Tensor, float, int]]]], prefix: str
    ):
        """Adds values as postfix string to progressbar.

        Args:
            prog_bar: a progressbar (on global rank zero) or an iterable (every other rank).
            candidates: the values to add as postfix strings to the progressbar.
            prefix: the prefix to add to each of these values.
        """
        if isinstance(prog_bar, tqdm) and candidates is not None:
            postfix_str = ""
            float_candidates = apply_to_collection(candidates, torch.Tensor, lambda x: x.item())
            if isinstance(candidates, torch.Tensor):
                postfix_str += f" {prefix}_loss: {float_candidates:.3f}"
            elif isinstance(candidates, Mapping):
                for k, v in float_candidates.items():
                    postfix_str += f" {prefix}_{k}: {v:.3f}"

            if postfix_str:
                prog_bar.set_postfix_str(postfix_str)

    def test(self, model: AnomalyModule, datamodule: AnomalibDataModule):
        """Runs the test routine.

        Args:
            model: The anomaly module.
            datamodule: The datamodule.
        """
        self.fabric.launch()

        if datamodule is not None:
            datamodule.prepare_data()
            datamodule.setup(stage="test")

        test_dataloader = model.test_dataloader() if datamodule is None else datamodule.test_dataloader()
        test_dataloader = self.fabric.setup_dataloaders(
            test_dataloader, use_distributed_sampler=self.use_distributed_sampler
        )

        # setup model and optimizer
        if isinstance(self.fabric.strategy, L.fabric.strategies.fsdp.FSDPStrategy):
            # currently, there is no way to support fsdp with model.configure_optimizers in fabric
            # as it would require fabric to hold a reference to the model, which we don't want to.
            raise NotImplementedError("BYOT currently does not support FSDP")

        self.fabric.call("on_test_start", trainer=self, pl_module=_unwrap_objects(model))
        model = self.fabric.setup(model)

        state = {"model": model}
        self.checkpoint_connector.restore(state)

        self.fabric.call("setup", trainer=self, pl_module=_unwrap_objects(model), stage="test")
        torch.set_grad_enabled(False)
        self.test_loop(model, test_dataloader, limit_batches=self.limit_test_batches)
        torch.set_grad_enabled(True)

    def test_loop(self, model: AnomalyModule, test_dataloader: torch.utils.data.DataLoader, limit_batches: int):
        self.fabric.call("on_test_epoch_start", trainer=self, pl_module=_unwrap_objects(model))
        iterable = self.progbar_wrapper(test_dataloader, total=min(len(test_dataloader), limit_batches), desc="Testing")
        for batch_idx, batch in enumerate(iterable):
            if self.should_stop or batch_idx >= limit_batches:
                self.fabric.call("on_test_epoch_end", trainer=self.fabric, pl_module=_unwrap_objects(model))
                return
            self.fabric.call(
                "on_test_batch_start",
                batch=batch,
                batch_idx=batch_idx,
                dataloader_idx=0,
                trainer=self.fabric,
                pl_module=_unwrap_objects(model),
            )
            out = model.test_step(batch, batch_idx)
            out = apply_to_collection(out, torch.Tensor, lambda x: x.detach())
            out = model.test_step_end(out)  # TODO change this

            self.fabric.call(
                "on_test_batch_end",
                outputs=out,
                batch=batch,
                batch_idx=batch_idx,
                dataloader_idx=0,
                trainer=self,
                pl_module=_unwrap_objects(model),
            )
            self._current_test_return.append(out)
        model.test_epoch_end(self._current_test_return)
        self.print_metrics(iterable, _unwrap_objects(model), prefix="test")
        self.fabric.call("on_test_epoch_end", trainer=self, pl_module=_unwrap_objects(model))
        self.fabric.call("on_test_model_train")
        self.fabric.call("teardown", trainer=self, pl_module=_unwrap_objects(model), stage="test")

    @property
    def test_dataloaders(self):
        return None  # TODO
