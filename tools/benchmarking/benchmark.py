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


import io
import os
import re
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Union

import pandas as pd
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from tqdm import tqdm

from anomalib.config import get_configurable_parameters
from anomalib.core.callbacks import get_callbacks
from anomalib.data import get_datamodule
from anomalib.models import get_model

MODEL_LIST = ["padim", "dfkde", "dfm", "patchcore", "stfpm"]
SEED = 42

# Modify category list according to dataset
CATEGORY_LIST = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


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


def get_single_model_metrics(model_name: str, gpu_count: int, category: str) -> Dict:
    """Collects metrics for `model_name` and returns a dict of results.

    Args:
        model_name (str): Name of the model
        gpu_count (int): Number of gpus. Use `gpu_count=0` for cpu
        category (str): Category of the dataset

    Returns:
        Dict: Collection of all the metrics such as time taken, throughput and performance scores.
    """
    config = get_configurable_parameters(model_name=model_name)
    # Seed for reproducibility
    seed_everything(42)

    config.trainer.gpus = gpu_count
    config.dataset.category = category

    with TemporaryDirectory() as project_path:
        config.project.path = project_path
        datamodule = get_datamodule(config)
        model = get_model(config)

        callbacks = update_callbacks(config)

        trainer = Trainer(**config.trainer, logger=None, callbacks=callbacks)

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            trainer.fit(model=model, datamodule=datamodule)

        # get training time
        captured_output = stdout.getvalue()
        pattern = re.compile(r"Training took (\d*.\d*) seconds")
        search_result = pattern.search(captured_output)
        training_time = float("nan")
        if search_result is not None:
            training_time = float(search_result.group(1))

        # Creating new variable is faster according to https://stackoverflow.com/a/4330829
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            trainer.test(model=model, datamodule=datamodule)

        captured_output = stdout.getvalue()

        # get testing time
        pattern = re.compile(r"Testing took (\d*.\d*) seconds")
        search_result = pattern.search(captured_output)
        testing_time = float("nan")
        if search_result is not None:
            testing_time = float(search_result.group(1))

        pattern = re.compile(r"Throughput\s?:\s?(\d+.\d+)")
        search_result = pattern.search(captured_output)
        throughput = float("nan")
        if search_result is not None:
            throughput = float(search_result.group(1))

        # Get metrics
        pattern = re.compile(r"\s?[\"'](\w+)[\"']:\s?(\d+.\d+)")
        metrics = re.findall(pattern, captured_output)

        # arrange the data
        data = {
            "Training Time (s)": training_time,
            "Testing Time (s)": testing_time,
            "Inference Throughput (fps)": throughput,
        }
        for key, val in metrics:
            data[key] = float(val)

    return data


def sweep():
    """Go over all models, categories, and devices and collect metrics."""
    for model_name in MODEL_LIST:
        metrics_list = []
        for category in tqdm(CATEGORY_LIST, desc=f"{model_name}:"):
            for gpu_count in range(0, 2):
                model_metrics = get_single_model_metrics(model_name, gpu_count, category)
                model_metrics["Device"] = "CPU" if gpu_count == 0 else "GPU"
                model_metrics["Category"] = category
                metrics_list.append(model_metrics)
        metrics_df = pd.DataFrame(metrics_list)
        result_path = Path(f"results/{model_name}.csv")
        os.makedirs(result_path.parent, exist_ok=True)
        metrics_df.to_csv(result_path)


if __name__ == "__main__":
    sweep()
