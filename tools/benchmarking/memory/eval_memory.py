"""Script called in subprocess to measure memory usage. """

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import csv
import sys
from pathlib import Path
from importlib import import_module

import tracemalloc
import psutil

import torch


def benchmark(train_fn, args, csv_file):
    # start tracking allocations
    tracemalloc.start()

    train_fn(args)

    # get peak allocated memory during run
    _, trml_peak = tracemalloc.get_traced_memory()
    trml_peak_mb = round(trml_peak / 10**6, 3)

    # get peak used memory during run
    cuda_peak = torch.cuda.max_memory_allocated()
    cuda_mb = round(cuda_peak / 10**6, 3)

    tracemalloc.stop()

    # get process peak memory (peak_wset work only on windows)
    p = psutil.Process()
    ps_peak = p.memory_info().peak_wset
    ps_peak_page = p.memory_info().peak_pagefile
    psutil_peak_mb = round(ps_peak / 10**6, 3)
    psutil_peak_page_mb = round(ps_peak_page / 10**6, 3)

    # write results to csv created by parent benchmark script
    with open(csv_file, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow([trml_peak_mb, cuda_mb, psutil_peak_mb, psutil_peak_page_mb])
        f.flush()


if __name__ == "__main__":
    # name of script that has train function we want to benchmark
    script_name = Path(sys.argv[1])
    # csv file where results will be appended
    output_csv = Path(sys.argv[2])

    # dynamical get train function and argparse function
    module = import_module(f"tools.{script_name.stem}")
    train_function = getattr(module, "train")
    arg_parser = getattr(module, "get_parser")

    # parse args specific to script and ignore additional script name
    args = arg_parser().parse_known_args()[0]

    benchmark(train_function, args, output_csv)
