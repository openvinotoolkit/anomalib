"""Benchmark all the algorithms in the repo."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import functools
import io
import logging
import math
import multiprocessing
import sys
import time
import warnings
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Union, cast

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from utils import upload_to_comet, upload_to_wandb, write_metrics

from anomalib.config import get_configurable_parameters, update_input_size_config
from anomalib.data import get_datamodule
from anomalib.deploy import export
from anomalib.deploy.export import ExportMode
from anomalib.models import get_model
from anomalib.utils.loggers import configure_logger
from anomalib.utils.sweep import (
    get_openvino_throughput,
    get_run_config,
    get_sweep_callbacks,
    get_torch_throughput,
    set_in_nested_config,
)

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
configure_logger()
pl_logger = logging.getLogger(__file__)
for logger_name in ["pytorch_lightning", "torchmetrics", "os"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)


def hide_output(func):
    """Decorator to hide output of the function.

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
        except Exception as exp:
            raise Exception(buf.getvalue()) from exp
        sys.stdout = std_out
        return value

    return wrapper


@hide_output
def get_single_model_metrics(model_config: Union[DictConfig, ListConfig], openvino_metrics: bool = False) -> Dict:
    """Collects metrics for `model_name` and returns a dict of results.

    Args:
        model_config (DictConfig, ListConfig): Configuration for run
        openvino_metrics (bool): If True, converts the model to OpenVINO format and gathers inference metrics.

    Returns:
        Dict: Collection of all the metrics such as time taken, throughput and performance scores.
    """

    with TemporaryDirectory() as project_path:
        model_config.project.path = project_path
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

        throughput = get_torch_throughput(model_config, model, datamodule.test_dataloader().dataset)

        # Get OpenVINO metrics
        openvino_throughput = float("nan")
        if openvino_metrics:
            # Create dirs for openvino model export
            openvino_export_path = project_path / Path("exported_models")
            openvino_export_path.mkdir(parents=True, exist_ok=True)
            export(model, model_config.model.input_size, ExportMode.OPENVINO, openvino_export_path)
            openvino_throughput = get_openvino_throughput(
                model_config, openvino_export_path, datamodule.test_dataloader().dataset
            )

        # arrange the data
        data = {
            "Training Time (s)": training_time,
            "Testing Time (s)": testing_time,
            "Inference Throughput (fps)": throughput,
            "OpenVINO Inference Throughput (fps)": openvino_throughput,
        }
        for key, val in test_results[0].items():
            data[key] = float(val)

    return data


def compute_on_cpu(sweep_config: Union[DictConfig, ListConfig], folder: Optional[str] = None):
    """Compute all run configurations over a sigle CPU."""
    for run_config in get_run_config(sweep_config.grid_search):
        model_metrics = sweep(run_config, 0, sweep_config.seed, False)
        write_metrics(model_metrics, sweep_config.writer, folder)


def compute_on_gpu(
    run_configs: List[DictConfig],
    device: int,
    seed: int,
    writers: List[str],
    folder: Optional[str] = None,
    compute_openvino: bool = False,
):
    """Go over each run config and collect the result.

    Args:
        run_configs (Union[DictConfig, ListConfig]): List of run configurations.
        device (int): The GPU id used for running the sweep.
        seed (int): Fix a seed.
        writers (List[str]): Destinations to write to.
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


def distribute_over_gpus(sweep_config: Union[DictConfig, ListConfig], folder: Optional[str] = None):
    """Distribute metric collection over all available GPUs. This is done by splitting the list of configurations."""
    with ProcessPoolExecutor(
        max_workers=torch.cuda.device_count(), mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        run_configs = list(get_run_config(sweep_config.grid_search))
        jobs = []
        for device_id, run_split in enumerate(
            range(0, len(run_configs), math.ceil(len(run_configs) / torch.cuda.device_count()))
        ):
            jobs.append(
                executor.submit(
                    compute_on_gpu,
                    run_configs[run_split : run_split + math.ceil(len(run_configs) / torch.cuda.device_count())],
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


def distribute(config: Union[DictConfig, ListConfig]):
    """Run all cpu experiments on a single process. Distribute gpu experiments over all available gpus.

    Args:
        config: (Union[DictConfig, ListConfig]): Sweep configuration.
    """

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
                except Exception as exception:
                    raise Exception(f"Error occurred while computing benchmark on device {job}") from exception
    elif "cpu" in devices:
        compute_on_cpu(config, folder=runs_folder)
    elif "gpu" in devices:
        distribute_over_gpus(config, folder=runs_folder)
    if "wandb" in config.writer:
        upload_to_wandb(team="anomalib", folder=runs_folder)
    if "comet" in config.writer:
        upload_to_comet(folder=runs_folder)


def sweep(
    run_config: Union[DictConfig, ListConfig], device: int = 0, seed: int = 42, convert_openvino: bool = False
) -> Dict[str, Union[float, str]]:
    """Go over all the values mentioned in `grid_search` parameter of the benchmarking config.

    Args:
        run_config: (Union[DictConfig, ListConfig], optional): Configuration for current run.
        device (int, optional): Name of the device on which the model is trained. Defaults to 0 "cpu".
        convert_openvino (bool, optional): Whether to convert the model to openvino format. Defaults to False.

    Returns:
        Dict[str, Union[float, str]]: Dictionary containing the metrics gathered from the sweep.
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
    else:
        model_config.trainer.accelerator = "cpu"

    # Remove legacy flags
    for legacy_device in ["num_processes", "gpus", "ipus", "tpu_cores"]:
        if legacy_device in model_config.trainer:
            model_config.trainer[legacy_device] = None

    if run_config.model_name in ["patchcore", "cflow"]:
        convert_openvino = False  # `torch.cdist` is not supported by onnx version 11
        # TODO Remove this line when issue #40 is fixed https://github.com/openvinotoolkit/anomalib/issues/40
        if model_config.model.input_size != (224, 224):
            return {}  # go to next run

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


if __name__ == "__main__":
    # Benchmarking entry point.
    # Spawn multiple processes one for cpu and rest for the number of gpus available in the system.
    # The idea is to distribute metrics collection over all the available devices.

    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, help="Path to sweep configuration")
    _args = parser.parse_args()

    print("Benchmarking started 🏃‍♂️. This will take a while ⏲ depending on your configuration.")
    _sweep_config = OmegaConf.load(_args.config)
    distribute(_sweep_config)
    print("Finished gathering results ⚡")
