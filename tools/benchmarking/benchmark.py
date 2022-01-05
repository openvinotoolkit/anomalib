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


import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Union, cast

import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from anomalib.config import get_configurable_parameters, update_input_size_config
from anomalib.core.callbacks import get_callbacks
from anomalib.core.model.anomaly_module import AnomalyModule
from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.utils.optimize import export_convert
from anomalib.utils.sweep import get_run_config, set_in_nested_config
from anomalib.utils.sweep.helpers.inference import (
    get_meta_data,
    get_openvino_throughput,
    get_torch_throughput,
)


def convert_to_openvino(model: AnomalyModule, export_path: Union[Path, str], input_size: List[int]):
    """Convert the trained model to OpenVINO."""
    export_path = export_path if isinstance(export_path, Path) else Path(export_path)
    onnx_path = export_path / "model.onnx"
    export_convert(model, input_size, onnx_path, export_path)


def update_callbacks(config: Union[DictConfig, ListConfig]) -> List[Callback]:
    """Disable callbacks in config.

    Args:
        config (Union[DictConfig, ListConfig]): Model config loaded from anomalib
    """
    config.project.log_images_to = []  # disable visualizer callback

    # disable openvino optimization
    if "optimization" in config.keys():
        config.pop("optimization")

    # disable load model callback
    if "weight_file" in config.model.keys():
        config.model.pop("weight_file")

    # disable save_to_csv. Metrics will be captured from stdout
    if "save_to_csv" in config.project.keys():
        config.project.pop("save_to_csv")

    callbacks = get_callbacks(config)

    # remove ModelCheckpoint callback
    for index, callback in enumerate(callbacks):
        if isinstance(callback, ModelCheckpoint):
            callbacks.pop(index)
            break

    return callbacks


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

        callbacks = update_callbacks(model_config)

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

        meta_data = get_meta_data(model, model_config.model.input_size)

        throughput = get_torch_throughput(model_config, model, datamodule.test_dataloader().dataset, meta_data)

        # Get OpenVINO metrics
        openvino_throughput = float("nan")
        if openvino_metrics:
            # Create dirs for openvino model export
            openvino_export_path = project_path / Path("exported_models")
            openvino_export_path.mkdir(parents=True, exist_ok=True)
            convert_to_openvino(model, openvino_export_path, model_config.model.input_size)
            openvino_throughput = get_openvino_throughput(
                model_config, openvino_export_path, datamodule.test_dataloader().dataset, meta_data
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


def sweep(device: str = "gpu"):
    """Go over all the values mentioned in ```grid_search``` parameter of the benchmarking config.

    Args:
        device (str, optional): Name of the device on which the model is trained. Defaults to "gpu".
    """
    sweep_config = OmegaConf.load("tools/benchmarking/benchmark_params.yaml")
    seed_everything(sweep_config.seed)

    for run_config in get_run_config(sweep_config.grid_search):
        # This assumes that ```model_name``` is always present in the sweep config.
        model_config = get_configurable_parameters(model_name=run_config.model_name)

        model_config = cast(DictConfig, model_config)  # placate mypy
        for param in run_config.keys():
            # grid search keys are always assumed to be strings
            param = cast(str, param)  # placate mypy
            set_in_nested_config(model_config, param.split("."), run_config[param])

        # convert image size to tuple in case it was updated by run config
        model_config = update_input_size_config(model_config)

        # Set device in config
        model_config.trainer.gpus = 1 if device == "gpu" else 0
        convert_openvino = bool(model_config.trainer.gpus)

        if run_config.model_name == "patchcore":
            convert_openvino = False  # ```torch.cdist```` is not supported by onnx version 11
            # TODO Remove this line when issue #40 is fixed https://github.com/openvinotoolkit/anomalib/issues/40
            if model_config.model.input_size != (224, 224):
                continue  # go to next run

        # Run benchmarking for current config
        model_metrics = get_single_model_metrics(model_config=model_config, openvino_metrics=convert_openvino)

        # Append configuration of current run to the collected metrics
        for key, value in run_config.items():
            # Skip adding model name to the dataframe
            if key != "model_name":
                model_metrics[key] = value

        # Add device name to list
        model_metrics["device"] = device

        # Write to file as each run is computed
        metrics_df = pd.DataFrame(model_metrics, index=[0])
        result_path = Path(f"results/{run_config.model_name}_{device}.csv")
        os.makedirs(result_path.parent, exist_ok=True)
        if not os.path.isfile(result_path):
            metrics_df.to_csv(result_path)
        else:
            metrics_df.to_csv(result_path, mode="a", header=False)


if __name__ == "__main__":
    print("Benchmarking started 🏃‍♂️. This will take a while ⏲ depending on your configuration.")
    # Spawn two processes one for cpu and one for gpu
    with ProcessPoolExecutor(max_workers=2) as executor:
        job = {executor.submit(sweep, device): device for device in ["gpu", "cpu"]}
        for hardware in as_completed(job):
            try:
                hardware.result()
            except Exception as exc:
                raise Exception(f"Error occurred while computing benchmark on device {job[hardware]}") from exc
    print("Finished gathering results ⚡")
