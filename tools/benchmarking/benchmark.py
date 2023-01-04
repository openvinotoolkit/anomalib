"""Benchmark all the algorithms in the repo."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from argparse import ArgumentParser
from pathlib import Path

from anomalib.utils.benchmark import distribute


def run(config_path: Path):
    """Run the benchmarking."""
    print("Benchmarking started 🏃‍♂️. This will take a while ⏲ depending on your configuration.")
    distribute(config_path)
    print("Finished gathering results ⚡")


if __name__ == "__main__":
    # Benchmarking entry point.
    # Spawn multiple processes one for cpu and rest for the number of gpus available in the system.
    # The idea is to distribute metrics collection over all the available devices.

    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, help="Path to sweep configuration")
    _args = parser.parse_args()

    run(_args.config)
