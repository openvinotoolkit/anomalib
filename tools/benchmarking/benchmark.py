"""Benchmark all the algorithms in the repo."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging
import warnings
from argparse import ArgumentParser
from pathlib import Path

from omegaconf import OmegaConf

from anomalib.utils.benchmarking import distribute

warnings.filterwarnings("ignore")
for logger_name in ["pytorch_lightning", "torchmetrics", "os"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)


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
