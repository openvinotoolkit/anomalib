"""Methods to compute and save metrics."""

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
import random
import string
import time
from glob import glob
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Union

import pandas as pd
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import Trainer
from torch.utils.tensorboard.writer import SummaryWriter

import wandb
from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.utils.sweep import (
    get_meta_data,
    get_openvino_throughput,
    get_sweep_callbacks,
    get_torch_throughput,
)

from .convert import convert_to_openvino


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

        callbacks = get_sweep_callbacks()

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


def write_metrics(model_metrics: Dict[str, Union[str, float]], writers: List[str]):
    """Writes metrics to destination provided in the sweep config.

    Args:
        model_metrics (Dict): Dictionary to be written
        writers (List[str]): List of destinations.
    """
    # Write to file as each run is computed
    if model_metrics == {} or model_metrics is None:
        return

    # Write to CSV
    metrics_df = pd.DataFrame(model_metrics, index=[0])
    result_path = Path(f"runs/{model_metrics['model_name']}_{model_metrics['device']}.csv")
    Path.mkdir(result_path.parent, parents=True, exist_ok=True)
    if not result_path.is_file():
        metrics_df.to_csv(result_path)
    else:
        metrics_df.to_csv(result_path, mode="a", header=False)

    if "tensorboard" in writers:
        write_to_tensorboard(model_metrics)


def write_to_tensorboard(
    model_metrics: Dict[str, Union[str, float]],
):
    """Write model_metrics to tensorboard.

    Args:
        model_metrics (Dict[str, Union[str, float]]): Dictionary containing collected results.
    """
    scalar_metrics = {}
    scalar_prefixes: List[str] = []
    string_metrics = {}
    for key, metric in model_metrics.items():
        if isinstance(metric, (int, float, bool)):
            scalar_metrics[key] = metric
        else:
            string_metrics[key] = metric
            scalar_prefixes.append(metric)
    writer = SummaryWriter(f"runs/{model_metrics['model_name']}_{model_metrics['device']}")
    for key, metric in model_metrics.items():
        if isinstance(metric, (int, float, bool)):
            scalar_metrics[key.replace(".", "/")] = metric  # need to join by / for tensorboard grouping
            writer.add_scalar(key, metric)
        else:
            if key == "model_name":
                continue
            scalar_prefixes.append(metric)
    scalar_prefix: str = "/".join(scalar_prefixes)
    for key, metric in scalar_metrics.items():
        writer.add_scalar(scalar_prefix + "/" + str(key), metric)
    writer.close()


def get_unique_key(str_len: int) -> str:
    """Returns a random string of length str_len.

    Args:
        str_len (int): Length of string.

    Returns:
        str: Random string
    """
    return "".join([random.choice(string.ascii_lowercase) for _ in range(str_len)])


def upload_to_wandb(team: str = "anomalib"):
    """Upload the data in csv files to wandb.

    Creates a project named benchmarking_[two random characters]. This is so that the project names are unique.
    One issue is that it does not check for collision

    Args:
        team (str, optional): Name of the team on wandb. This can also be the id of your personal account.
        Defaults to "anomalib".
    """
    project = f"benchmarking_{get_unique_key(2)}"
    tag_list = ["dataset.category", "model_name", "dataset.image_size", "model.backbone", "device"]
    for csv_file in glob("runs/*.csv"):
        table = pd.read_csv(csv_file)
        for index, row in table.iterrows():
            row = dict(row[1:])  # remove index column
            tags = [str(row[column]) for column in tag_list if column in row.keys()]
            wandb.init(
                entity=team, project=project, name=f"{row['model_name']}_{row['dataset.category']}_{index}", tags=tags
            )
            wandb.log(row)
            wandb.finish()
