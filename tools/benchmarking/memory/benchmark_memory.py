"""Functions used to benchmark memory consumption of models."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import subprocess
import shutil
from pathlib import Path
import csv
from argparse import ArgumentParser

from datetime import datetime
import time

import logging

from tqdm import tqdm

from anomalib.utils.loggers import configure_logger

logger = logging.getLogger(__name__)
configure_logger()
pl_logger = logging.getLogger(__file__)
for logger_name in ["pytorch_lightning", "torchmetrics", "os"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)


def memory_arg_parser() -> ArgumentParser:
    """Get parser for memory args.

    Returns:
        ArgumentParser: The parser object.
    """
    parser = ArgumentParser()
    parser.add_argument("--one_config", type=str, required=True, help="Path to a one model config file")
    parser.add_argument("--ens_config", type=str, required=True, help="Path to a ensemble of models config file")
    parser.add_argument("--number", type=int, required=True, help="Number of repetitions")

    return parser


def run_mem_benchmark(args):
    # create directory for current run
    time_stamp = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    parent_dir = Path(f"tools/benchmarking/memory/runs/memory_run_{time_stamp}")
    parent_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "one": {"train_name": "train.py", "config": Path(args.one_config)},
        "ens": {"train_name": "train_ensemble.py", "config": Path(args.ens_config)},
    }
    for model, properties in models.items():
        logger.info(f"Running memory benchmark for {model}.")

        # copy config to current dir
        shutil.copy(properties["config"], parent_dir)

        result_file_path = parent_dir / f"{model}.csv"

        # args used for subprocess call
        call_args = [
            "python",
            "tools/benchmarking/memory/eval_memory.py",
            properties["train_name"],
            result_file_path,
            "--config",
            str(properties["config"]),
            "--log-level",
            "ERROR",
        ]

        # create csv file and write header
        with open(result_file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(["tracemalloc", "cuda", "psutil_peak", "psutil_page_peak"])

        # repeat measurement for provided number of times
        for _ in tqdm(range(args.number)):
            try:
                # call fresh process where training is done and logged to csv
                subprocess.run(call_args, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Run failed: {e}")
                return


if __name__ == "__main__":
    script_args = memory_arg_parser().parse_args()
    run_mem_benchmark(script_args)
