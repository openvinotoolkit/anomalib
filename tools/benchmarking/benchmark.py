"""Benchmark all the algorithms in the repo."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path

# E402 Module level import not at top of file. Disabled as we need to redirect all outputs during the runs.
# ruff: noqa: E402


# File cannot be unique because if we create a unique name based on time,
# each process will create a new file
log_file = "runs/benchmark.log"
Path(log_file).parent.mkdir(exist_ok=True, parents=True)
logger_file_handler = logging.FileHandler(log_file)
logger_file_handler.setLevel(logging.INFO)

# Redirect warnings and logs to file that are generated while importing
logging.captureWarnings(True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logger_file_handler)

import functools
import io
import math
import multiprocessing
import sys
import time
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from enum import Enum

# End of warnings capture | Rest of the imports follow
from multiprocessing.managers import DictProxy
from tempfile import TemporaryDirectory
from typing import cast

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from rich import print
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table
from utils import upload_to_comet, upload_to_wandb, write_metrics

from anomalib.config import get_configurable_parameters, update_input_size_config
from anomalib.data import get_datamodule
from anomalib.deploy import export
from anomalib.deploy.export import ExportMode
from anomalib.models import get_model
from anomalib.utils.sweep import get_run_config, get_sweep_callbacks, set_in_nested_config

# TODO add torch and openvino throughputs.

# Redirect future warnings and logs to file from all the imports
loggers = [name for name in logging.root.manager.loggerDict if ("lightning" in name or "anomalib" in name)]
for logger_name in loggers + ["py.warnings"]:
    _logger = logging.getLogger(logger_name)
    _logger.setLevel(logging.WARNING)
    _logger.handlers = []
    _logger.addHandler(logger_file_handler)


class Status(str, Enum):
    """Status of the benchmarking run."""

    SUCCESS = "success"
    FAILED = "failed"


def redirect_output(func):
    """Decorator to redirect output of the function.

    Args:
        func (function): Hides output of this function.

    Raises:
        Exception: Incase the execution of function fails, it raises an exception.

    Returns:
        object of the called function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        std_out = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            value = func(*args, **kwargs)
            logger.info(buf.getvalue())
            logger.info(value)
        except Exception as exp:
            logger.exception(
                f"Error occurred while computing benchmark {exp}. Buffer: {buf.getvalue()}."
                f"\n Method {func}, args {args}, kwargs {kwargs}"
            )
            value = ""
        sys.stdout = std_out
        return value

    return wrapper


@redirect_output
def get_single_model_metrics(model_config: DictConfig | ListConfig) -> dict:
    """Collects metrics for `model_name` and returns a dict of results.

    Args:
        model_config (DictConfig, ListConfig): Configuration for run

    Returns:
        dict: Collection of all the metrics such as time taken, throughput and performance scores.
    """

    with TemporaryDirectory() as project_path:
        model_config.project.path = project_path
        model_config.trainer.enable_progress_bar = False
        model_config.trainer.enable_model_summary = False
        datamodule = get_datamodule(model_config)
        model = get_model(model_config)

        callbacks = get_sweep_callbacks(model_config)

        trainer = Trainer(**model_config.trainer, logger=None, callbacks=callbacks)

        start_time = time.time()

        trainer.fit(model=model, datamodule=datamodule)

        # get start time
        training_time = time.time() - start_time

        # Creating new variable is faster according to https://stackoverflow.com/a/4330829
        start_time = time.time()
        # get test results
        test_results = trainer.test(model=model, datamodule=datamodule)

        # get testing time
        testing_time = time.time() - start_time

        # Create dirs for torch export (as default only lighting model is produced)
        export(
            task=model_config.dataset.task,
            transform=trainer.datamodule.test_data.transform.to_dict(),
            input_size=model_config.model.input_size,
            model=model,
            export_mode=ExportMode.TORCH,
            export_root=project_path,
        )

        # arrange the data
        data = {
            "Training Time (s)": training_time,
            "Testing Time (s)": testing_time,
        }
        for key, val in test_results[0].items():
            data[key] = float(val)

    return data


def compute_on_gpu(
    run_configs: list[DictConfig],
    device: int,
    seed: int,
    writers: list[str],
    folder: str,
    compute_openvino: bool = False,
):
    """Go over each run config and collect the result.

    Args:
        run_configs (DictConfig | ListConfig): List of run configurations.
        device (int): The GPU id used for running the sweep.
        seed (int): Fix a seed.
        writers (list[str]): Destinations to write to.
        folder (optional, str): Sub-directory to which runs are written to. Defaults to None. If none writes to root.
        compute_openvino (bool, optional): Compute OpenVINO throughput. Defaults to False.
    """
    for run_config in run_configs:
        if isinstance(run_config, (DictConfig, ListConfig)):
            model_metrics = sweep(run_config, device, seed, compute_openvino)
            write_metrics(model_metrics, writers, folder)
        else:
            raise ValueError(
                f"Expecting `run_config` of type DictConfig or ListConfig. Got {type(run_config)} instead."
            )


def distribute_over_gpus(sweep_config: DictConfig | ListConfig, folder: str):
    """Distribute metric collection over all available GPUs. This is done by splitting the list of configurations."""
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as executor:
        run_configs = list(get_run_config(sweep_config.grid_search))
        jobs = []
        for device_id, run_split in enumerate(range(0, len(run_configs), math.ceil(len(run_configs) / 1))):
            jobs.append(
                executor.submit(
                    compute_on_gpu,
                    run_configs[run_split : run_split + math.ceil(len(run_configs) / 1)],
                    device_id + 1,
                    sweep_config.seed,
                    sweep_config.writer,
                    folder,
                    sweep_config.compute_openvino,
                )
            )
        for job in jobs:
            try:
                job.result()
            except Exception as exc:
                raise Exception(f"Error occurred while computing benchmark on GPU {job}") from exc


def sweep(
    run_config: DictConfig | ListConfig, device: int = 0, seed: int = 42, convert_openvino: bool = False
) -> dict[str, str | float]:
    """Go over all the values mentioned in `grid_search` parameter of the benchmarking config.

    Args:
        run_config: (DictConfig | ListConfig, optional): Configuration for current run.
        device (int, optional): Name of the device on which the model is trained. Defaults to 0 "cpu".
        convert_openvino (bool, optional): Whether to convert the model to openvino format. Defaults to False.

    Returns:
        dict[str, str | float]: Dictionary containing the metrics gathered from the sweep.
    """
    seed_everything(seed, workers=True)
    # This assumes that `model_name` is always present in the sweep config.
    model_config = get_configurable_parameters(model_name=run_config.model_name)
    model_config.project.seed = seed

    model_config = cast(DictConfig, model_config)  # placate mypy
    for param in run_config.keys():
        # grid search keys are always assumed to be strings
        param = cast(str, param)  # placate mypy
        set_in_nested_config(model_config, param.split("."), run_config[param])  # type: ignore

    # convert image size to tuple in case it was updated by run config
    model_config = update_input_size_config(model_config)

    # Set device in config. 0 - cpu, [0], [1].. - gpu id
    if device != 0:
        model_config.trainer.devices = [device - 1]
        model_config.trainer.accelerator = "gpu"

    # Remove legacy flags
    for legacy_device in ["num_processes", "gpus", "ipus", "tpu_cores"]:
        if legacy_device in model_config.trainer:
            model_config.trainer[legacy_device] = None

    # Run benchmarking for current config
    model_metrics = get_single_model_metrics(model_config=model_config, openvino_metrics=convert_openvino)
    output = f"One sweep run complete for model {model_config.model.name}"
    output += f" On category {model_config.dataset.category}" if model_config.dataset.category is not None else ""
    output += str(model_metrics)
    logger.info(output)

    # Append configuration of current run to the collected metrics
    for key, value in run_config.items():
        # Skip adding model name to the dataframe
        if key != "model_name":
            model_metrics[key] = value

    # Add device name to list
    model_metrics["device"] = "gpu" if device > 0 else "cpu"
    model_metrics["model_name"] = run_config.model_name

    return model_metrics


class Benchmark:
    """Benchmarking runner

    Args:
        config: (DictConfig | ListConfig): Sweep configuration.
        n_gpus: (int): Number of devices to run the benchmark on. If n_gpus is 0, benchmarking is run on all available
         GPUs.
    """

    def __init__(self, config: DictConfig | ListConfig, n_gpus: int = 0):
        self.config = config
        self.n_gpus = min(n_gpus, torch.cuda.device_count()) if n_gpus > 0 else torch.cuda.device_count()
        self.runs_folder = f"runs/{datetime.strftime(datetime.now(), '%Y_%m_%d-%H_%M_%S')}"
        Path(self.runs_folder).mkdir(exist_ok=True, parents=True)

    def _sweep(self, device: int, run_config: DictConfig, seed: int = 42) -> dict:
        """Run a single sweep on a device."""
        seed_everything(seed, workers=True)
        # This assumes that `model_name` is always present in the sweep config.
        model_config = get_configurable_parameters(model_name=run_config.model_name)
        model_config.project.seed = seed
        model_config = cast(DictConfig, model_config)  # placate mypy
        for param in run_config.keys():
            # grid search keys are always assumed to be strings
            param = cast(str, param)  # placate mypy
            set_in_nested_config(model_config, param.split("."), run_config[param])  # type: ignore

        # convert image size to tuple in case it was updated by run config
        model_config = update_input_size_config(model_config)

        # Set device in config. 0 - cpu, [0], [1].. - gpu id
        if device != 0:
            model_config.trainer.devices = [device - 1]
            model_config.trainer.accelerator = "gpu"

        # Remove legacy flags
        for legacy_device in ["num_processes", "gpus", "ipus", "tpu_cores"]:
            if legacy_device in model_config.trainer:
                model_config.trainer[legacy_device] = None

        # Run benchmarking for current config
        model_metrics = get_single_model_metrics(model_config=model_config)
        output = f"One sweep run complete for model {model_config.model.name}"
        output += f" On category {model_config.dataset.category}" if model_config.dataset.category is not None else ""
        output += str(model_metrics)
        logger.info(output)

        # Append configuration of current run to the collected metrics
        for key, value in run_config.items():
            # Skip adding model name to the dataframe
            if key != "model_name":
                model_metrics[key] = value

        # Add device name to list
        model_metrics["device"] = "gpu"
        model_metrics["model_name"] = run_config.model_name

        return model_metrics

    def _compute(
        self, progress: DictProxy, task_id: TaskID, device: int, run_configs: list[DictConfig]
    ) -> dict[str, list[str]]:
        """Iterate over configurations and compute & write metrics for single configuration.

        Args:
            progress (DictProxy): Shared dict to write progress status for displaying in terminal.
            task_id (TaskID): Task id for the current process. Used to identify the progress bar.
            device (int): GPU id on which the benchmarking is run.
            run_configs (list[DictConfig]): List of run configurations.

        Returns:
            dict[str, list[str]]: Dictionary containing the metrics gathered from the sweep.
        """
        result = []
        for idx, config in enumerate(run_configs):
            try:
                output = self._sweep(device, config)
                write_metrics(output, self.config.writer, self.runs_folder)
                result.append(output)

                progress[str(task_id)] = {"completed": idx + 1, "total": len(run_configs)}
            except Exception as exception:
                logger.exception(
                    f"Error occurred while computing benchmark on GPU {device} with config {config}, {exception}"
                    f"\nLocals {locals()}"
                )
        # convert list of dicts to dict of lists
        return {key: [dic[key] for dic in result] for key in result[0]}

    def _distribute(self) -> Status:
        run_configs = list(get_run_config(self.config.grid_search))
        step_size = math.ceil(len(run_configs) / self.n_gpus)
        jobs = []
        results: list[dict[str, list[str]]] = []
        status = Status.SUCCESS
        with Progress() as progress:
            overall_progress_task = progress.add_task("[green]Overall Progress")
            with multiprocessing.Manager() as manager:
                _progress = manager.dict()

                with ProcessPoolExecutor(
                    max_workers=self.n_gpus, mp_context=multiprocessing.get_context("spawn")
                ) as executor:
                    for device_id, run_split in enumerate(range(0, len(run_configs), step_size)):
                        task_id = progress.add_task(f"Running benchmark on GPU {device_id}")
                        _progress[str(task_id)] = {"completed": 0, "total": step_size}
                        jobs.append(
                            executor.submit(
                                self._compute,
                                _progress,
                                task_id,
                                device_id,
                                run_configs[run_split : run_split + step_size],
                            )
                        )

                    # monitor the progress:
                    while (n_finished := sum([job.done() for job in jobs])) < len(jobs):
                        progress.update(overall_progress_task, completed=n_finished, total=self.n_gpus)
                        for task_id, params in _progress.items():
                            progress.update(TaskID(int(task_id)), completed=params["completed"], total=params["total"])

                    for job in jobs:
                        try:
                            results.append(job.result())
                        except Exception as exception:
                            logger.exception(
                                f"Error occurred while collecting benchmark {exception}\nLocals {locals()}"
                            )

                    try:
                        progress.update(overall_progress_task, completed=self.n_gpus, total=self.n_gpus)
                        result: dict[str, list] = {key: [] for key in results[0].keys()}
                        for _result in results:
                            for key, value in _result.items():
                                result[key].extend(value)
                    except Exception as exception:
                        logger.exception(
                            f"Error occurred while merging benchmark results {exception}\nResult {results}"
                        )
                        status = Status.FAILED

        try:
            console = Console()
            table = Table(title="Benchmarking Results", show_header=True, header_style="bold magenta")
            for column in result.keys():
                table.add_column(column)
            for row in [*zip(*result.values())]:
                table.add_row(*[str(value) for value in row])
            console.print(table)
        except Exception as exception:
            logger.exception(f"Error occurred while printing benchmarking results {exception}. Result: {result}")
            status = Status.FAILED
        return status

    def run(self):
        """Run the benchmarking."""
        logger.info(
            f"\n{'-'*120}\n"
            f"Starting benchmarking. {datetime.strftime(datetime.now(), '%Y %m %d-%H %M %S')}"
            f"\nDistributing benchmark collection over {self.n_gpus} GPUs."
        )
        if not torch.cuda.is_available():
            logger.warning("Could not detect any cuda enabled devices")
        if self._distribute() != Status.SUCCESS:
            print(
                "[bold red]There were some errors while collecting benchmark[/bold red]"
                f"\nPlease check the log file [magenta]{self.runs_folder}/benchmark.log[/magenta]"
                " for more details."
            )
        else:
            if "wandb" in self.config.writer:
                upload_to_wandb(team="anomalib", folder=self.runs_folder)
            if "comet" in self.config.writer:
                upload_to_comet(folder=self.runs_folder)
        logger.info("Benchmarking complete" f"\n{'-'*120}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, help="Path to sweep configuration")
    _args = parser.parse_args()

    print("[royal_blue1]Benchmarking started. This will take a while depending on your configuration.[/royal_blue1]")
    _sweep_config = OmegaConf.load(_args.config)
    runner = Benchmark(_sweep_config, 0)
    runner.run()
    print("[royal_blue1]Finished gathering results[/royal_blue1] ⚡")
