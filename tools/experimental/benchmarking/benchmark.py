"""Run benchmarking."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from anomalib.pipelines.benchmark import Benchmark

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.warning("This feature is experimental. It may change or be removed in the future.")
    Benchmark().run()
