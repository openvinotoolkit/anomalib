"""Methods to compute and save metrics."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import random
import string
from pathlib import Path

import pandas as pd
from comet_ml import Experiment
from torch.utils.tensorboard.writer import SummaryWriter

import wandb

logger = logging.getLogger(__name__)


def write_metrics(
    model_metrics: dict[str, str | float],
    writers: list[str],
    folder: str,
):
    """Writes metrics to destination provided in the sweep config.

    Args:
        model_metrics (dict): Dictionary to be written
        writers (list[str]): List of destinations.
        folder (optional, str): Sub-directory to which runs are written to. Defaults to None. If none writes to root.
    """
    # Write to file as each run is computed
    if model_metrics == {} or model_metrics is None:
        return

    result_folder = Path(folder)
    # Write to CSV
    try:
        metrics_df = pd.DataFrame(model_metrics, index=[0])
        result_path = result_folder / f"{model_metrics['model_name']}_{model_metrics['device']}.csv"
        Path.mkdir(result_path.parent, parents=True, exist_ok=True)
        if not result_path.is_file():
            metrics_df.to_csv(result_path)
        else:
            metrics_df.to_csv(result_path, mode="a", header=False)
    except Exception as exception:
        logger.exception(f"Could not write to csv. Exception: {exception}")

    project_name = f"benchmarking_{result_folder.name}"
    tags = []
    for key, value in model_metrics.items():
        if all(name not in key.lower() for name in ["time", "image", "pixel", "throughput"]):
            tags.append(str(value))

    if "tensorboard" in writers:
        write_to_tensorboard(model_metrics, result_folder)
    if "wandb" in writers:
        write_to_wandb(model_metrics, project_name, tags)
    if "comet" in writers:
        write_to_comet(model_metrics, project_name, tags)


def write_to_tensorboard(
    model_metrics: dict[str, str | float],
    folder: Path,
):
    """Write model_metrics to tensorboard.

    Args:
        model_metrics (dict[str, str | float]): Dictionary containing collected results.
    """
    scalar_metrics = {}
    scalar_prefixes: list[str] = []
    string_metrics = {}
    for key, metric in model_metrics.items():
        if isinstance(metric, (int, float, bool)):
            scalar_metrics[key] = metric
        else:
            string_metrics[key] = metric
            scalar_prefixes.append(metric)
    writer = SummaryWriter(folder / "tfevents")
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
    return "".join([random.choice(string.ascii_lowercase) for _ in range(str_len)])  # nosec: B311


def write_to_wandb(
    model_metrics: dict[str, str | float],
    project_name: str,
    tags: list[str],
    team: str = "anomalib",
):
    """Write model_metrics to wandb.

    > _Note:_ It is observed that any failure in wandb causes the run to hang. Use wandb writer with caution.

    Args:
        model_metrics (dict[str, str | float]): Dictionary containing collected results.
        project_name (str): Name of the project on wandb.
        tags (list[str]): List of tags for the run.
        team (str, optional): Name of the team on wandb. This can also be the id of your personal account.
            Defaults to "anomalib".
    """
    for key, value in model_metrics.items():
        if all(name not in key.lower() for name in ["time", "image", "pixel", "throughput"]):
            tags.append(str(value))
    run = wandb.init(
        entity=team,
        project=project_name,
        name=f"{'_'.join(tags)}",
        tags=tags,
        settings={"silent": True, "show_info": False, "show_warnings": False, "show_errors": False},
    )
    run.log(model_metrics)
    logger.info(f"Run logged at {run.url}")
    run.finish(quiet=True)


def write_to_comet(
    model_metrics: dict[str, str | float],
    project_name: str,
    tags: list[str],
    team: str = "anomalib",
):
    """Write model_metrics to wandb.


    Args:
        model_metrics (dict[str, str | float]): Dictionary containing collected results.
        project_name (str): Name of the project on comet.
        tags (list[str]): List of tags for the run.
        team (str, optional): Name of the team on wandb. This can also be the id of your personal account.
            Defaults to "anomalib".
    """
    experiment = Experiment(project_name=project_name, workspace=team)
    experiment.set_name(f"{'_'.join(tags)}")
    experiment.log_metrics(model_metrics, step=1, epoch=1)  # populates auto-generated charts on panel view
    experiment.add_tags(tags)
    logger.info(f"Run logged at {experiment.url}")
    experiment.end()
