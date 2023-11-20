"""Benchmark all the algorithms in the repo."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from argparse import ArgumentParser
from pathlib import Path

from anomalib.pipelines.benchmarking import distribute

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, help="Path to sweep configuration")
    _args = parser.parse_args()

    print("Benchmarking started 🏃‍♂️. This will take a while ⏲ depending on your configuration.")
    distribute(_args.config)
    print("Finished gathering results ⚡")
