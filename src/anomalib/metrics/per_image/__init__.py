"""Per-Image Metrics."""

# Original Code
# https://github.com/jpcbertoldo/aupimo
#
# Modified
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .binclf_curve_numpy import BinclfThreshsChoice
from .pimo import AUPIMO, PIMO, AUPIMOResult, PIMOResult, aupimo_scores, pimo_curves
from .utils import (
    compare_models_pairwise_ttest_rel,
    compare_models_pairwise_wilcoxon,
    format_pairwise_tests_results,
    per_image_scores_stats,
)
from .utils_numpy import StatsOutliersPolicy, StatsRepeatedPolicy

__all__ = [
    # constants
    "BinclfThreshsChoice",
    "StatsOutliersPolicy",
    "StatsRepeatedPolicy",
    # result classes
    "PIMOResult",
    "AUPIMOResult",
    # functional interfaces
    "pimo_curves",
    "aupimo_scores",
    # torchmetrics interfaces
    "PIMO",
    "AUPIMO",
    # utils
    "compare_models_pairwise_ttest_rel",
    "compare_models_pairwise_wilcoxon",
    "format_pairwise_tests_results",
    "per_image_scores_stats",
]
