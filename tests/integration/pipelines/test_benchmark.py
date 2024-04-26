"""Test benchmarking pipeline."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

from anomalib.pipelines import Benchmark


@patch("anomalib.pipelines.benchmark.job.Engine", return_value=MagicMock(test=MagicMock(return_value=[{"test": 1}])))
@patch("anomalib.pipelines.benchmark.generator.get_model", return_value=MagicMock())
@patch("anomalib.pipelines.benchmark.generator.get_datamodule", return_value=MagicMock(category="dummy"))
def test_benchmark_pipeline(engine: MagicMock, model: MagicMock, datamodule: MagicMock) -> None:  # noqa: ARG001 | this is needed for patching
    """Test benchmarking pipeline."""
    with patch("anomalib.pipelines.benchmark.job.BenchmarkJob.save", return_value=MagicMock()) as save_method:
        benchmark = Benchmark()
        benchmark_parser = benchmark.get_parser()
        args = benchmark_parser.parse_args(["--config", "tests/integration/pipelines/pipeline.yaml"])
        benchmark.run(args)
        assert len(save_method.call_args.args[0]) == 8
