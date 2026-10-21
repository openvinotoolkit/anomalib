# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test benchmarking pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from anomalib.pipelines import Benchmark


@patch(
    "anomalib.pipelines.benchmark.job.Engine",
    return_value=MagicMock(test=MagicMock(return_value=[{"test": 1}]), fit=MagicMock(return_value=None)),
)
@patch("anomalib.pipelines.benchmark.generator.get_model", return_value=MagicMock())
@patch("anomalib.pipelines.benchmark.generator.get_datamodule")
def test_benchmark_pipeline(mock_datamodule: MagicMock, model: MagicMock, engine: MagicMock) -> None:  # noqa: ARG001
    """Test benchmarking pipeline."""
    # Setup mock datamodule and loader
    mock_dataloader = MagicMock()
    mock_dataloader.dataset = [1] * 10
    mock_dataloader.batch_size = 2
    
    mock_dm_instance = MagicMock(category="dummy")
    mock_dm_instance.test_dataloader.return_value = [mock_dataloader]
    mock_datamodule.return_value = mock_dm_instance

    with patch("anomalib.pipelines.benchmark.job.BenchmarkJob.save", return_value=MagicMock()) as save_method, \
         patch("time.perf_counter", side_effect=[1.0, 2.0, 3.0, 4.0] * 100), \
         patch("time.time", side_effect=[float(i) for i in range(1000)]):
        
        benchmark = Benchmark()
        benchmark_parser = benchmark.get_parser()
        config_path = Path(__file__).parent / "pipeline.yaml"
        args = benchmark_parser.parse_args(["--config", str(config_path)])
        
        # Override inference enabled config
        benchmark.run(args)
        
        assert len(save_method.call_args.args[0]) == 8
        saved_df = save_method.call_args.args[0]
        assert "job_duration" in saved_df.columns
        assert "fit_duration" in saved_df.columns
        assert "test_duration" in saved_df.columns
        assert "test_throughput_fps" in saved_df.columns
        assert "test_num_images" in saved_df.columns

        # Verify inference bench logic
        import yaml
        import tempfile
        config_text = config_path.read_text()
        config_dict = yaml.safe_load(config_text)
        if "benchmark" in config_dict and "inference" in config_dict["benchmark"]:
            config_dict["benchmark"]["inference"]["enabled"] = True
        
        with tempfile.TemporaryDirectory() as td:
            inference_config_path = Path(td) / "pipeline_inference.yaml"
            inference_config_path.write_text(yaml.dump(config_dict))
            
            args_inference = benchmark_parser.parse_args(["--config", str(inference_config_path)])
            benchmark.run(args_inference)
            assert len(save_method.call_args.args[0]) == 8
