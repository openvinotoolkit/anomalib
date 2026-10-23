# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatible benchmark job arguments and heterogeneous result collection."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from anomalib.pipelines.benchmark.job import BenchmarkJob


def test_original_positional_arguments() -> None:
    """Existing public callers need not supply the newly optional trainer arguments."""
    job = BenchmarkJob("cpu", MagicMock(), MagicMock(), 42, {})
    assert job.trainer_params == {}


@pytest.mark.parametrize("key", ["accelerator", "devices", "default_root_dir"])
def test_reserved_trainer_arguments(key: str) -> None:
    """Job-owned device/directory arguments cannot conflict with trainer overrides."""
    with pytest.raises(ValueError, match="Benchmark job controls"):
        BenchmarkJob("cpu", MagicMock(), MagicMock(), 42, {}, trainer_params={key: "invalid"})


def test_optional_metric_columns() -> None:
    """Collection preserves image-only results alongside segmentation results."""
    results = BenchmarkJob.collect([{"image_AUROC": 0.9}, {"image_AUROC": 0.8, "pixel_AUROC": 0.7}])
    assert len(results) == 2
    assert pd.isna(results.iloc[0]["pixel_AUROC"])
