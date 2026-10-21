# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Benchmarking job for evaluating model performance.

This module provides functionality for running individual benchmarking jobs that
evaluate model performance on specific datasets. Each job runs a model on a dataset
and collects performance metrics.

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models import Padim
    >>> from anomalib.pipelines.benchmark.job import BenchmarkJob

    >>> # Initialize model, datamodule and job
    >>> model = Padim()
    >>> datamodule = MVTecAD(category="bottle")
    >>> job = BenchmarkJob(
    ...     accelerator="gpu",
    ...     model=model,
    ...     datamodule=datamodule,
    ...     seed=42,
    ...     flat_cfg={"model.name": "padim"}
    ... )

    >>> # Run the benchmark job
    >>> results = job.run()

The job executes model training and evaluation, collecting metrics like accuracy,
F1-score, and inference time. Results are returned in a standardized format for
comparison across different model-dataset combinations.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
import torch
from lightning import seed_everything
from rich.console import Console
from rich.table import Table

from anomalib.data import AnomalibDataModule
from anomalib.deploy import ExportType, OpenVINOInferencer, TorchInferencer
from anomalib.engine import Engine
from anomalib.models import AnomalibModule
from anomalib.pipelines.components import Job
from anomalib.pipelines.benchmark.utils import (
    extract_images_from_batch,
    get_device_from_model,
    get_test_dataloader,
)
from anomalib.utils.logging import hide_output

try:
    from torch.utils.flop_counter import FlopTensorDispatchMode
except ImportError:
    FlopTensorDispatchMode = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)


class BenchmarkJob(Job):
    """Benchmarking job for evaluating anomaly detection models.

    This class implements a benchmarking job that evaluates model performance by
    training and testing on a given dataset. It collects metrics like accuracy,
    F1-score, and timing information.

    Args:
        accelerator (str): Type of accelerator to use for computation (e.g.
            ``"cpu"``, ``"gpu"``).
        model (AnomalibModule): Anomaly detection model instance to benchmark.
        datamodule (AnomalibDataModule): Data module providing the dataset.
        seed (int): Random seed for reproducibility.
        flat_cfg (dict): Flattened configuration dictionary with dotted keys.

    Example:
        >>> from anomalib.data import MVTecAD
        >>> from anomalib.models import Padim
        >>> from anomalib.pipelines.benchmark.job import BenchmarkJob

        >>> # Initialize model, datamodule and job
        >>> model = Padim()
        >>> datamodule = MVTecAD(category="bottle")
        >>> job = BenchmarkJob(
        ...     accelerator="gpu",
        ...     model=model,
        ...     datamodule=datamodule,
        ...     seed=42,
        ...     flat_cfg={"model.name": "padim"}
        ... )

        >>> # Run the benchmark job
        >>> results = job.run()

    The job executes model training and evaluation, collecting metrics like
    accuracy, F1-score, and inference time. Results are returned in a standardized
    format for comparison across different model-dataset combinations.
    """

    name = "benchmark"

    def __init__(
        self,
        accelerator: str,
        model: AnomalibModule,
        datamodule: AnomalibDataModule,
        seed: int,
        flat_cfg: dict,
    ) -> None:
        super().__init__()
        self.accelerator = accelerator
        self.model = model
        self.datamodule = datamodule
        self.seed = seed
        self.flat_cfg = flat_cfg

    @hide_output
    def run(
        self,
        task_id: int | None = None,
    ) -> dict[str, Any]:
        """Run the benchmark job.

        This method executes the full benchmarking pipeline including model
        training and testing. It measures execution time for different stages and
        collects performance metrics.

        Args:
            task_id (int | None, optional): ID of the task when running in
                distributed mode. When provided, the job will use the specified
                device. Defaults to ``None``.

        Returns:
            dict[str, Any]: Dictionary containing benchmark results including:
                - Timing information (job, fit and test duration)
                - Model configuration
                - Performance metrics from testing
        """
        # Config knobs for optional inference micro-benchmarking.
        # Note: `flat_cfg` is produced by flattening the `benchmark:` config section,
        # so users typically specify `inference.*` under `benchmark:`.
        inference_bench_enabled = bool(
            self.flat_cfg.get(
                "inference.enabled", False
            ),
        )
        inference_warmup_runs = int(
            self.flat_cfg.get(
                "inference.warmup_runs", 10
            ),
        )
        inference_runs = int(
            self.flat_cfg.get(
                "inference.runs", 50
            ),
        )
        inference_batch_size = self.flat_cfg.get(
            "inference.batch_size", None
        )

        job_start_time = time.time()
        devices: str | list[int] = "auto"
        if task_id is not None:
            devices = [task_id]
            logger.info(f"Running job {self.model.__class__.__name__} with device {task_id}")
        with TemporaryDirectory() as temp_dir:
            seed_everything(self.seed)
            engine = Engine(
                accelerator=self.accelerator,
                devices=devices,
                default_root_dir=temp_dir,
            )
            fit_start_time = time.time()
            engine.fit(self.model, self.datamodule)

            # Optional micro-benchmark: inferencer throughput.
            inference_bench: dict[str, Any] = {}
            if inference_bench_enabled:
                inference_bench = self._benchmark_inference_throughput(
                    engine=engine,
                    warmup_runs=inference_warmup_runs,
                    runs=inference_runs,
                    batch_size=inference_batch_size,
                    temp_dir=Path(temp_dir),
                )

            test_start_time = time.time()
            test_results = engine.test(self.model, self.datamodule)
            test_end_time = time.time()
        job_end_time = time.time()
        durations = {
            "job_duration": job_end_time - job_start_time,
            "fit_duration": test_start_time - fit_start_time,
            "test_duration": test_end_time - test_start_time,
        }

        # Restore throughput in benchmark outputs.
        # https://github.com/open-edge-platform/anomalib/issues/2054
        test_loader_info = self._get_test_loader_info()
        throughput: dict[str, Any] = {}
        if test_loader_info["num_images"] is not None and durations["test_duration"] > 0:
            throughput = {
                "test_num_images": test_loader_info["num_images"],
                "test_batch_size": test_loader_info["batch_size"],
                "test_throughput_fps": test_loader_info["num_images"] / durations["test_duration"],
            }

        output = {
            "accelerator": self.accelerator,
            **durations,
            **throughput,
            **inference_bench,
            **self.flat_cfg,
            **test_results[0],
        }
        logger.info(f"Completed with result {output}")
        return output

    def _get_test_loader_info(self) -> dict[str, int | None]:
        """Best-effort test loader metadata.

        Returns:
            dict[str, int | None]: Keys:
                - num_images: Total number of images in the test dataset, if available.
                - batch_size: Batch size of the test dataloader, if available.
        """
        try:
            dataloaders = self.datamodule.test_dataloader()
        except Exception:  # noqa: BLE001 - best-effort only
            return {"num_images": None, "batch_size": None}

        dataloader = dataloaders[0] if isinstance(dataloaders, (list, tuple)) else dataloaders
        num_images: int | None
        batch_size: int | None
        try:
            num_images = len(dataloader.dataset)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - best-effort only
            num_images = None
        batch_size = getattr(dataloader, "batch_size", None)
        return {"num_images": num_images, "batch_size": batch_size}

    def _yield_batches(self, dataloader: Any) -> Any:  # noqa: ANN401
        """Yield batches from the dataloader infinitely."""
        while True:
            yield from dataloader

    def _benchmark_inferencer(
        self,
        inferencer: Any,  # noqa: ANN401
        dataloader: Any,  # noqa: ANN401
        device: torch.device,
        warmup_runs: int,
        runs: int,
        count_flops: bool = False,
    ) -> dict[str, Any]:
        """Benchmark an inferencer implementation."""
        batch_iter = self._yield_batches(dataloader)
        
        def _get_next_images() -> tuple[Any, int]:  # noqa: ANN401
            batch = next(batch_iter)
            images = extract_images_from_batch(batch)
            if images is None:
                return None, 0
            return images, int(images.shape[0])

        with torch.inference_mode():
            for _ in range(max(0, warmup_runs)):
                images, _ = _get_next_images()
                if images is None:
                    break
                inferencer.predict(images)
            
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elif device.type == "mps":
                torch.mps.synchronize()

            flops_per_run: float | None = None
            if count_flops:
                try:
                    images, _ = _get_next_images()
                    if FlopTensorDispatchMode is not None and images is not None and hasattr(inferencer, "model"):
                        with FlopTensorDispatchMode(inferencer.model) as flop_counter:
                            inferencer.predict(images)
                        counts = flop_counter.flop_counts.get("Global", {})
                        flops_per_run = sum(counts.values()) if counts else None
                except Exception:  # noqa: BLE001
                    pass

        total_images = 0
        with torch.inference_mode():
            start_time = time.perf_counter()
            for _ in range(max(1, runs)):
                images, num_images = _get_next_images()
                if images is None:
                    break
                inferencer.predict(images)
                total_images += num_images

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elif device.type == "mps":
                torch.mps.synchronize()
            
            duration = time.perf_counter() - start_time
            
        return {
            "duration": duration,
            "num_images": total_images,
            "flops_per_run": flops_per_run,
        }

    def _benchmark_inference_throughput(  # noqa: C901
        self,
        engine: Engine,
        warmup_runs: int,
        runs: int,
        batch_size: int | None,
        temp_dir: Path,
    ) -> dict[str, Any]:
        """Benchmark Torch and OpenVINO end-to-end throughput."""
        metrics: dict[str, Any] = {}
        try:
            dataloader = get_test_dataloader(self.datamodule, batch_size)
            if dataloader is None:
                return {}

            device = get_device_from_model(self.model)

            # Torch Benchmark
            try:
                logger.info("Exporting to Torch format for throughput benchmarking.")
                torch_model_path = engine.export(
                    model=self.model,
                    export_type=ExportType.TORCH,
                    export_root=temp_dir / "export",
                    model_file_name="benchmark_torch",
                )
                if torch_model_path is not None:
                    inferencer = TorchInferencer(path=torch_model_path, device=device.type)
                    res = self._benchmark_inferencer(
                        inferencer, dataloader, device, warmup_runs, runs, count_flops=True
                    )
                    if res["num_images"] > 0:
                        metrics["torch_num_images"] = res["num_images"]
                        metrics["torch_duration"] = res["duration"]
                        metrics["torch_throughput_fps"] = res["num_images"] / res["duration"]
                        if res["flops_per_run"]:
                            metrics["torch_flops"] = res["flops_per_run"]
            except Exception as ex:  # noqa: BLE001
                logger.warning(f"Failed to benchmark Torch model throughput: {ex}")

            # OpenVINO Benchmark
            try:
                logger.info("Exporting to OpenVINO format for throughput benchmarking.")
                ov_model_path = engine.export(
                    model=self.model,
                    export_type=ExportType.OPENVINO,
                    export_root=temp_dir / "export",
                    model_file_name="benchmark_ov",
                    datamodule=self.datamodule,
                )
                if ov_model_path is not None:
                    ov_device = "GPU" if device.type == "cuda" else "CPU"
                    inferencer = OpenVINOInferencer(path=ov_model_path, device=ov_device)
                    res = self._benchmark_inferencer(
                        inferencer, dataloader, device, warmup_runs, runs, count_flops=False
                    )
                    if res["num_images"] > 0:
                        metrics["openvino_num_images"] = res["num_images"]
                        metrics["openvino_duration"] = res["duration"]
                        metrics["openvino_throughput_fps"] = res["num_images"] / res["duration"]
            except Exception as ex:  # noqa: BLE001
                logger.warning(f"Failed to benchmark OpenVINO model throughput: {ex}")

        except Exception as ex:  # noqa: BLE001
            logger.warning(f"Inference throughput benchmarking failed: {ex}")
            
        return metrics

    @staticmethod
    def collect(results: list[dict[str, Any]]) -> pd.DataFrame:
        """Collect and aggregate results from multiple benchmark runs.

        Args:
            results (list[dict[str, Any]]): List of result dictionaries from
                individual benchmark runs.

        Returns:
            pd.DataFrame: DataFrame containing aggregated results with each row
                representing a benchmark run.
        """
        if not results:
            return pd.DataFrame()

        output: dict[str, Any] = {}
        for key in results[0]:
            output[key] = []
        for result in results:
            for key, value in result.items():
                output[key].append(value)
        return pd.DataFrame(output)

    @staticmethod
    def save(result: pd.DataFrame) -> None:
        """Save benchmark results to CSV file.

        The results are saved in the ``runs/benchmark/YYYY-MM-DD-HH_MM_SS``
        directory. The method also prints a tabular view of the results.

        Args:
            result (pd.DataFrame): DataFrame containing benchmark results to save.
        """
        BenchmarkJob._print_tabular_results(result)
        file_path = Path("runs") / BenchmarkJob.name / datetime.now().strftime("%Y-%m-%d-%H_%M_%S") / "results.csv"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(file_path, index=False)
        logger.info(f"Saved results to {file_path}")

    @staticmethod
    def _print_tabular_results(gathered_result: pd.DataFrame) -> None:
        """Print benchmark results in a formatted table.

        Args:
            gathered_result (pd.DataFrame): DataFrame containing results to
                display.
        """
        if gathered_result is not None:
            console = Console()
            table = Table(title=f"{BenchmarkJob.name} Results", show_header=True, header_style="bold magenta")
            results = gathered_result.to_dict("list")
            for column in results:
                table.add_column(column)
            for row in zip(*results.values(), strict=False):
                table.add_row(*[str(value) for value in row])
            console.print(table)
