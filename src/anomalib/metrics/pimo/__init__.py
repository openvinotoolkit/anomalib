"""Per-Image Metrics."""

# Original Code
# https://github.com/jpcbertoldo/aupimo
#
# Modified
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .binary_classification_curve import ThresholdMethod
from .pimo import AUPIMO, PIMO, AUPIMOResult, PIMOResult
from .utils_benchmark import (
    AUPIMO_BENCHMARK_DATASETS,
    AUPIMO_BENCHMARK_MODELS,
    aupimo_result_from_json_dict,
    aupimo_result_to_json_dict,
    download_aupimo_benchmark_scores,
    get_aupimo_benchmark,
)

__all__ = [
    # constants
    "ThresholdMethod",
    # result classes
    "PIMOResult",
    "AUPIMOResult",
    # torchmetrics interfaces
    "PIMO",
    "AUPIMO",
    "StatsOutliersPolicy",
    # utils_benchmark
    "AUPIMO_BENCHMARK_DATASETS",
    "AUPIMO_BENCHMARK_MODELS",
    "aupimo_result_from_json_dict",
    "aupimo_result_to_json_dict",
    "download_aupimo_benchmark_scores",
    "get_aupimo_benchmark",
]
