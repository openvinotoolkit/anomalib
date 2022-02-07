"""Benchmark all the algorithms in the repo."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.


import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Union, cast

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import seed_everything
from utils import get_single_model_metrics, write_metrics

from anomalib.config import get_configurable_parameters, update_input_size_config
from anomalib.utils.sweep import get_run_config, set_in_nested_config


def compute_over_cpu():
    """Compute all run configurations over a sigle CPU."""
    sweep_config = OmegaConf.load("tools/benchmarking/benchmark_params.yaml")
    for run_config in get_run_config(sweep_config.grid_search):
        model_metrics = sweep(run_config, 0, sweep_config.seed)
        write_metrics(model_metrics, sweep_config.writer)


def compute_over_gpu(run_configs: Union[DictConfig, ListConfig], device: int, seed: int, writers: List[str]):
    """Go over each run config and collect the result.

    Args:
        run_configs (Union[DictConfig, ListConfig]): List of run configurations.
        device (int): The GPU id used for running the sweep.
        seed (int): Fix a seed.
        writers (List[str]): Destinations to write to.
    """
    for run_config in run_configs:
        model_metrics = sweep(run_config, device, seed)
        write_metrics(model_metrics, writers)


def distribute_over_gpus():
    """Distribute metric collection over all available GPUs. This is done by splitting the list of configurations."""
    sweep_config = OmegaConf.load("tools/benchmarking/benchmark_params.yaml")
    with ProcessPoolExecutor(
        max_workers=torch.cuda.device_count(), mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        run_configs = list(get_run_config(sweep_config.grid_search))
        jobs = []
        for device_id, run_split in enumerate(
            range(0, len(run_configs), len(run_configs) // torch.cuda.device_count())
        ):
            jobs.append(
                executor.submit(
                    compute_over_gpu,
                    run_configs[run_split : run_split + len(run_configs) // torch.cuda.device_count()],
                    device_id + 1,
                    sweep_config.seed,
                    sweep_config.writer,
                )
            )
        for job in jobs:
            try:
                job.result()
            except Exception as exc:
                raise Exception(f"Error occurred while computing benchmark on device {job}") from exc


def distribute():
    """Run all cpu experiments on a single process. Distribute gpu experiments over all available gpus.

    Args:
        device_count (int, optional): If device count is 0, uses only cpu else spawn processes according
        to number of gpus available on the machine. Defaults to 0.
    """
    if not torch.cuda.is_available():
        compute_over_cpu()
    else:
        # Create process for gpu and cpu
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as executor:
            jobs = [executor.submit(compute_over_cpu), executor.submit(distribute_over_gpus)]
            for job in as_completed(jobs):
                try:
                    job.result()
                except Exception as exception:
                    raise Exception(f"Error occurred while computing benchmark on device {job}") from exception


def sweep(run_config: Union[DictConfig, ListConfig], device: int = 0, seed: int = 42) -> Dict[str, Union[float, str]]:
    """Go over all the values mentioned in `grid_search` parameter of the benchmarking config.

    Args:
        device (int, optional): Name of the device on which the model is trained. Defaults to 0 "cpu".

    Returns:
        Dict[str, Union[float, str]]: Dictionary containing the metrics gathered from the sweep.
    """
    seed_everything(seed)
    # This assumes that `model_name` is always present in the sweep config.
    model_config = get_configurable_parameters(model_name=run_config.model_name)

    model_config = cast(DictConfig, model_config)  # placate mypy
    for param in run_config.keys():
        # grid search keys are always assumed to be strings
        param = cast(str, param)  # placate mypy
        set_in_nested_config(model_config, param.split("."), run_config[param])

    # convert image size to tuple in case it was updated by run config
    model_config = update_input_size_config(model_config)

    # Set device in config. 0 - cpu, [0], [1].. - gpu id
    model_config.trainer.gpus = 0 if device == 0 else [device - 1]
    convert_openvino = bool(model_config.trainer.gpus)

    if run_config.model_name == "patchcore":
        convert_openvino = False  # `torch.cdist` is not supported by onnx version 11
        # TODO Remove this line when issue #40 is fixed https://github.com/openvinotoolkit/anomalib/issues/40
        if model_config.model.input_size != (224, 224):
            return {}  # go to next run

    # Run benchmarking for current config
    model_metrics = get_single_model_metrics(model_config=model_config, openvino_metrics=convert_openvino)

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

    print("Benchmarking started 🏃‍♂️. This will take a while ⏲ depending on your configuration.")
    distribute()
    print("Finished gathering results ⚡")
