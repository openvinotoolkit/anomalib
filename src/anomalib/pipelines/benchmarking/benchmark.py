"""Benchmark all the algorithms in the repo."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
import io
import logging
import math
import multiprocessing
import sys
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from lightning.pytorch import seed_everything
from omegaconf import DictConfig, ListConfig, OmegaConf

from anomalib.callbacks.timer import TimerCallback
from anomalib.data import get_datamodule
from anomalib.deploy.export import export_to_openvino, export_to_torch
from anomalib.engine import Engine
from anomalib.loggers import configure_logger
from anomalib.models import get_model
from anomalib.pipelines.sweep import get_openvino_throughput, get_run_config, get_torch_throughput
from anomalib.pipelines.sweep.config import flattened_config_to_nested
from anomalib.utils.config import update_input_size_config

from .utils import upload_to_comet, upload_to_wandb, write_metrics

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
configure_logger()
pl_logger = logging.getLogger(__file__)
for logger_name in ["lightning.pytorch", "torchmetrics", "os"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)


def hide_output(func: Callable[..., Any]) -> Callable[..., Any]:
    """Hide output of the function.

    Args:
        func (function): Hides output of this function.

    Raises:
        Exception: In case the execution of function fails, it raises an exception.

    Returns:
        object of the called function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:  # noqa: ANN401
        std_out = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            value = func(*args, **kwargs)
        # NOTE: A generic exception is used here to catch all exceptions.
        except Exception as exception:  # noqa: BLE001
            raise Exception(buf.getvalue()) from exception  # noqa: TRY002
        sys.stdout = std_out
        return value

    return wrapper


@hide_output
def get_single_model_metrics(
    accelerator: str,
    devices: int | list[int],
    model_config: DictConfig | ListConfig,
    openvino_metrics: bool = False,
) -> dict:
    """Collect metrics for `model_name` and returns a dict of results.

    Args:
        accelerator (str): The device on which the model is trained. "cpu" or "gpu".
        devices (int | list[int]): The GPU id used for running the sweep. This is used to select a particular GPU on
            a multi-gpu system.
        model_config (DictConfig, ListConfig): Configuration for run
        openvino_metrics (bool): If True, converts the model to OpenVINO format and gathers inference metrics.

    Returns:
        dict: Collection of all the metrics such as time taken, throughput and performance scores.
    """
    with TemporaryDirectory() as project_path:
        datamodule = get_datamodule(model_config)
        model = get_model(model_config.model)

        engine = Engine(
            accelerator=accelerator,
            devices=devices,
            logger=None,
            callbacks=[TimerCallback()],
            default_root_dir=project_path,
        )

        start_time = time.time()

        engine.fit(model=model, datamodule=datamodule)

        # get start time
        training_time = time.time() - start_time

        # Creating new variable is faster according to https://stackoverflow.com/a/4330829
        start_time = time.time()
        # get test results
        test_results = engine.test(model=model, datamodule=datamodule)

        # get testing time
        testing_time = time.time() - start_time

        export_to_torch(
            model=model,
            export_root=Path(project_path),
            transform=engine.trainer.datamodule.test_data.transform,
            task=engine.trainer.datamodule.test_data.task,
        )

        throughput = get_torch_throughput(
            model_path=project_path,
            test_dataset=datamodule.test_dataloader().dataset,
            device=accelerator,
        )

        # Get OpenVINO metrics
        openvino_throughput = float("nan")
        if openvino_metrics:
            if "input_size" in model_config.model.init_args:
                input_size = model_config.model.init_args.input_size
            else:
                input_size = model_config.data.init_args.image_size
            export_to_openvino(
                export_root=Path(project_path),
                model=model,
                input_size=input_size,
                transform=engine.trainer.datamodule.test_data.transform,
                ov_args={},
                task=engine.trainer.datamodule.test_data.task,
            )
            openvino_throughput = get_openvino_throughput(model_path=project_path, test_dataset=datamodule.test_data)

        # arrange the data
        data = {
            "Training Time (s)": training_time,
            "Testing Time (s)": testing_time,
            f"Inference Throughput {accelerator} (fps)": throughput,
            "OpenVINO Inference Throughput (fps)": openvino_throughput,
        }
        for key, val in test_results[0].items():
            data[key] = float(val)

    return data


def compute_on_cpu(sweep_config: DictConfig | ListConfig, folder: str | None = None) -> None:
    """Compute all run configurations over a sigle CPU."""
    for run_config in get_run_config(sweep_config.grid_search):
        model_metrics = sweep(
            run_config=run_config,
            device=0,
            seed=sweep_config.seed_everything,
            convert_openvino=False,
        )
        write_metrics(model_metrics, sweep_config.writer, folder)


def compute_on_gpu(
    run_configs: list[DictConfig],
    device: int,
    seed: int,
    writers: list[str],
    folder: str | None = None,
    compute_openvino: bool = False,
) -> None:
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
        if isinstance(run_config, DictConfig | ListConfig):
            model_metrics = sweep(run_config=run_config, device=device, seed=seed, convert_openvino=compute_openvino)
            write_metrics(model_metrics, writers, folder)
        else:
            msg = f"Expecting `run_config` of type DictConfig or ListConfig. Got {type(run_config)} instead."
            raise TypeError(msg)


def distribute_over_gpus(sweep_config: DictConfig | ListConfig, folder: str | None = None) -> None:
    """Distribute metric collection over all available GPUs. This is done by splitting the list of configurations."""
    with ProcessPoolExecutor(
        max_workers=torch.cuda.device_count(),
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        run_configs = list(get_run_config(sweep_config.grid_search))
        jobs = []
        num_gpus = torch.cuda.device_count()
        chunk_size = math.ceil(len(run_configs) / num_gpus)
        for device_id, run_split in enumerate(range(0, len(run_configs), chunk_size)):
            jobs.append(
                executor.submit(
                    compute_on_gpu,
                    run_configs[run_split : run_split + chunk_size],
                    device_id + 1,
                    sweep_config.seed_everything,
                    sweep_config.writer,
                    folder,
                    sweep_config.compute_openvino,
                ),
            )
        for job in jobs:
            try:
                job.result()
            # NOTE: A generic exception is used here to catch all exceptions.
            except Exception as exception:  # noqa: BLE001, PERF203
                msg = f"Error occurred while computing benchmark on GPU {job}"
                raise Exception(msg) from exception  # noqa: TRY002


def distribute(config_path: Path) -> None:
    """Run all cpu experiments on a single process. Distribute gpu experiments over all available gpus.

    Args:
        config_path: (Path): Config path.
    """
    config = OmegaConf.load(config_path)
    runs_folder = datetime.strftime(datetime.now(), "%Y_%m_%d-%H_%M_%S")
    devices = config.hardware
    if not torch.cuda.is_available() and "gpu" in devices:
        pl_logger.warning("Config requested GPU benchmarking but torch could not detect any cuda enabled devices")
    elif {"cpu", "gpu"}.issubset(devices):
        # Create process for gpu and cpu
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as executor:
            jobs = [
                executor.submit(compute_on_cpu, config, runs_folder),
                executor.submit(distribute_over_gpus, config, runs_folder),
            ]
            for job in as_completed(jobs):
                try:
                    job.result()
                # NOTE: A generic exception is used here to catch all exceptions.
                except Exception as exception:  # noqa: BLE001, PERF203
                    msg = f"Error occurred while computing benchmark on device {job}"
                    raise Exception(msg) from exception  # noqa: TRY002
    elif "cpu" in devices:
        compute_on_cpu(config, folder=runs_folder)
    elif "gpu" in devices:
        distribute_over_gpus(config, folder=runs_folder)
    if "wandb" in config.writer:
        upload_to_wandb(team="anomalib", folder=runs_folder)
    if "comet" in config.writer:
        upload_to_comet(folder=runs_folder)


def sweep(
    run_config: DictConfig | ListConfig,
    device: int = 0,
    seed: int = 42,
    convert_openvino: bool = False,
) -> dict[str, str | float]:
    """Go over all the values mentioned in `grid_search` parameter of the benchmarking config.

    Args:
        run_config: (DictConfig | ListConfig, optional): Configuration for current run.
        device (int, optional): Name of the device on which the model is trained. Defaults to 0 "cpu".
        seed (int, optional): Seed to be used for the run. Defaults to 42.
        convert_openvino (bool, optional): Whether to convert the model to openvino format. Defaults to False.

    Returns:
        dict[str, str | float]: Dictionary containing the metrics gathered from the sweep.
    """
    seed_everything(seed, workers=True)

    model_config = flattened_config_to_nested(run_config)
    # Add model key if it does not exist
    # Model config needs to be created so that input size of the model can be updated based on the
    # data configuration.
    if "model" not in model_config:
        model_class = get_model(model_config.model_name).__class__
        model_config["model"] = model_config.get(
            "model",
            OmegaConf.create(
                {
                    "class_path": model_class.__module__ + "." + model_class.__name__,
                    "init_args": {
                        key: value.default
                        for key, value in inspect.signature(model_class).parameters.items()
                        if key != "self"
                    },
                },
            ),
        )
    model_config = update_input_size_config(model_config)

    # Set device in config. 0 - cpu, [0], [1].. - gpu id
    devices: list[int] | int
    if device != 0:
        devices = [device - 1]
        accelerator = "gpu"
    else:
        accelerator = "cpu"
        devices = device

    # Run benchmarking for current config
    model_metrics = get_single_model_metrics(
        accelerator=accelerator,
        devices=devices,
        model_config=model_config,
        openvino_metrics=convert_openvino,
    )
    output = f"One sweep run complete for model {model_config.model_name}"
    output += (
        f" On category {model_config.data.init_args.category}"
        if ("init_args" in model_config.data and "category" in model_config.data.init_args)
        else ""
    )
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
