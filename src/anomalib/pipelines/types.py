"""Types used in pipeline components.

This module defines type aliases used throughout the pipeline components for type
hinting and documentation.

The following types are defined:
    - ``RUN_RESULTS``: Return type of individual job runs
    - ``GATHERED_RESULTS``: Combined results from multiple job runs
    - ``PREV_STAGE_RESULT``: Optional results from previous pipeline stage

Example:
    >>> from anomalib.pipelines.types import RUN_RESULTS, GATHERED_RESULTS
    >>> def my_job() -> RUN_RESULTS:
    ...     return {"metric": 0.95}
    >>> def gather_results(results: list[RUN_RESULTS]) -> GATHERED_RESULTS:
    ...     return {"mean_metric": sum(r["metric"] for r in results) / len(results)}
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any

RUN_RESULTS = Any
GATHERED_RESULTS = Any
PREV_STAGE_RESULT = GATHERED_RESULTS | None
