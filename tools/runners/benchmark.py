"""Run benchmarking."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from anomalib.pipelines import Pipeline, PoolExecutor
from anomalib.pipelines.jobs import BenchmarkJob

if __name__ == "__main__":
    Pipeline(PoolExecutor(BenchmarkJob())).run()
