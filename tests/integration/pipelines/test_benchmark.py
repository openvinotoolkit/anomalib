# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test benchmarking pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from anomalib import __version__
from anomalib.engine import Engine
from anomalib.pipelines import Benchmark
from anomalib.pipelines.benchmark.job import BenchmarkJob
from tests.integration.test_task_types import DummyClassificationModel


@patch("anomalib.pipelines.benchmark.job.Engine", return_value=MagicMock(test=MagicMock(return_value=[{"test": 1}])))
@patch("anomalib.pipelines.benchmark.generator.get_model", return_value=MagicMock())
@patch("anomalib.pipelines.benchmark.generator.get_datamodule", return_value=MagicMock(category="dummy"))
@pytest.mark.parametrize("trainer", [None, {"max_epochs": 2}])
def test_benchmark_pipeline(
    datamodule: MagicMock,
    model: MagicMock,
    engine: MagicMock,
    tmp_path: Path,
    trainer: dict | None,
) -> None:
    """Old configs work without trainer settings; configured limits reach Engine."""
    config = yaml.safe_load(Path("tests/integration/pipelines/pipeline.yaml").read_text(encoding="utf-8"))
    if trainer is not None:
        config["benchmark"]["trainer"] = trainer
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(yaml.safe_dump(config))
    with patch("anomalib.pipelines.benchmark.job.BenchmarkJob.save", return_value=MagicMock()) as save_method:
        benchmark = Benchmark()
        benchmark_parser = benchmark.get_parser()
        args = benchmark_parser.parse_args(["--config", str(config_path)])
        benchmark.run(args)
        results = save_method.call_args.args[0]
        assert len(results) == 8
        assert model.call_count == datamodule.call_count == 8
        assert "anomalib_version" in results.columns
        assert (results["anomalib_version"] == __version__).all()
        assert engine.call_args.kwargs.get("max_epochs") == (trainer or {}).get("max_epochs")


def test_collect_image_only_results() -> None:
    """Image-only and segmentation results can share a CSV without inventing scores."""
    results = BenchmarkJob.collect([{"image_AUROC": 0.9}, {"image_AUROC": 0.8, "pixel_AUROC": 0.7}])
    assert len(results) == 2
    assert pd.isna(results.iloc[0]["pixel_AUROC"])


def test_nightly_trainer_setup(tmp_path: Path) -> None:
    """Nightly defaults construct a real Trainer compatible with Engine's callbacks."""
    matrix = yaml.safe_load(Path(".github/scripts/benchmark/matrix.yaml").read_text(encoding="utf-8"))
    engine = Engine(accelerator="cpu", devices=1, default_root_dir=tmp_path, **matrix["trainer"])
    engine._setup_trainer(DummyClassificationModel())  # noqa: SLF001 - exercise real Trainer construction
    assert engine.checkpoint_callback is not None
