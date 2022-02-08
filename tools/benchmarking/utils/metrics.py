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
from glob import glob
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
from torch.utils.tensorboard.writer import SummaryWriter

import wandb


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
