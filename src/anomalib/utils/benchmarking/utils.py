"""Methods to compute and save metrics."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import string
from pathlib import Path

import numpy as np
import pandas as pd

from anomalib.utils.exceptions import try_import

if try_import("wandb"):
    import wandb
if try_import("comet_ml"):
    from comet_ml import Experiment
if try_import("torch.utils.tensorboard.writer"):
    from torch.utils.tensorboard.writer import SummaryWriter


def write_metrics(
    model_metrics: dict[str, str | float],
    writers: list[str],
    folder: str | None = None,
) -> None:
    """Write metrics to destination provided in the sweep config.

    Args:
        model_metrics (dict): Dictionary to be written
        writers (list[str]): List of destinations.
        folder (optional, str): Sub-directory to which runs are written to. Defaults to None. If none writes to root.
    """
    # Write to file as each run is computed
    if model_metrics == {} or model_metrics is None:
        return

    # Write to CSV
    metrics_df = pd.DataFrame(model_metrics, index=[0])
    result_folder = Path("runs") if folder is None else Path(f"runs/{folder}")
    result_path = result_folder / f"{model_metrics['model_name']}_{model_metrics['device']}.csv"
    Path.mkdir(result_path.parent, parents=True, exist_ok=True)
    if not result_path.is_file():
        metrics_df.to_csv(result_path)
    else:
        metrics_df.to_csv(result_path, mode="a", header=False)

    if "tensorboard" in writers:
        write_to_tensorboard(model_metrics)


def write_to_tensorboard(
    model_metrics: dict[str, str | float],
) -> None:
    """Write model_metrics to tensorboard.

    Args:
        model_metrics (dict[str, str | float]): Dictionary containing collected results.
    """
    scalar_metrics = {}
    scalar_prefixes: list[str] = []
    string_metrics = {}
    for key, metric in model_metrics.items():
        if isinstance(metric, int | float | bool):
            scalar_metrics[key] = metric
        else:
            string_metrics[key] = metric
            scalar_prefixes.append(metric)
    writer = SummaryWriter(f"runs/{model_metrics['model_name']}_{model_metrics['device']}")
    for key, metric in model_metrics.items():
        if isinstance(metric, int | float | bool):
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
    """Return a random string of length str_len.

    Args:
        str_len (int): Length of string.

    Returns:
        str: Random string
    """
    return "".join([np.random.default_rng().choice(string.ascii_lowercase) for _ in range(str_len)])


def upload_to_wandb(
    team: str = "anomalib",
    folder: str | None = None,
) -> None:
    """Upload the data in csv files to wandb.

    Creates a project named benchmarking_[two random characters]. This is so that the project names are unique.
    One issue is that it does not check for collision

    Args:
        team (str, optional): Name of the team on wandb. This can also be the id of your personal account.
        Defaults to "anomalib".
        folder (optional, str): Sub-directory from which runs are picked up. Defaults to None. If none picks from runs.
    """
    project = f"benchmarking_{get_unique_key(2)}"
    tag_list = ["dataset.category", "model_name", "dataset.image_size", "model.backbone", "device"]
    search_path = "runs/*.csv" if folder is None else f"runs/{folder}/*.csv"
    for csv_file in Path(search_path).glob("*csv"):
        table = pd.read_csv(csv_file)
        for index, row_with_index_column in table.iterrows():
            row = dict(row_with_index_column[1:])  # remove index column
            tags = [str(row[column]) for column in tag_list if column in row]
            wandb.init(
                entity=team,
                project=project,
                name=f"{row['model_name']}_{row['dataset.category']}_{index}",
                tags=tags,
            )
            wandb.log(row)
            wandb.finish()


def upload_to_comet(
    folder: str | None = None,
) -> None:
    """Upload the data in csv files to comet.

    Creates a project named benchmarking_[two random characters]. This is so that the project names are unique.
    One issue is that it does not check for collision

    Args:
        folder (optional, str): Sub-directory from which runs are picked up. Defaults to None. If none picks from runs.
    """
    project = f"benchmarking_{get_unique_key(2)}"
    tag_list = ["dataset.category", "model_name", "dataset.image_size", "model.backbone", "device"]
    search_path = "runs/*.csv" if folder is None else f"runs/{folder}/*.csv"
    for csv_file in Path(search_path).glob("*csv"):
        table = pd.read_csv(csv_file)
        for index, row_with_index_column in table.iterrows():
            row = dict(row_with_index_column[1:])  # remove index column
            tags = [str(row[column]) for column in tag_list if column in row]
            experiment = Experiment(project_name=project)
            experiment.set_name(f"{row['model_name']}_{row['dataset.category']}_{index}")
            experiment.log_metrics(row, step=1, epoch=1)  # populates auto-generated charts on panel view
            experiment.add_tags(tags)
            experiment.log_table(filename=csv_file)
