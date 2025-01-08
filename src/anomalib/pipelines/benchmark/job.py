"""Benchmarking job for evaluating model performance.

This module provides functionality for running individual benchmarking jobs that
evaluate model performance on specific datasets. Each job runs a model on a dataset
and collects performance metrics.

Example:
    >>> from anomalib.data import MVTec
    >>> from anomalib.models import Padim
    >>> from anomalib.pipelines.benchmark.job import BenchmarkJob

    >>> # Initialize model, datamodule and job
    >>> model = Padim()
    >>> datamodule = MVTec(category="bottle")
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

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
from lightning import seed_everything
from rich.console import Console
from rich.table import Table

from anomalib.data import AnomalibDataModule
from anomalib.engine import Engine
from anomalib.models import AnomalibModule
from anomalib.pipelines.components import Job
from anomalib.utils.logging import hide_output

logger = logging.getLogger(__name__)

# Import external loggers
try:
    from anomalib.loggers import (
        AnomalibCometLogger,
        AnomalibMLFlowLogger,
        AnomalibTensorBoardLogger,
        AnomalibWandbLogger,
    )

    LOGGERS_AVAILABLE = True
    logger.info("Successfully imported logger modules.")
except ImportError:
    LOGGERS_AVAILABLE = False
    logger.warning("To use external loggers, install required packages using `anomalib install -v`")


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
        >>> from anomalib.data import MVTec
        >>> from anomalib.models import Padim
        >>> from anomalib.pipelines.benchmark.job import BenchmarkJob

        >>> # Initialize model, datamodule and job
        >>> model = Padim()
        >>> datamodule = MVTec(category="bottle")
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
                logger=self._initialize_loggers(self.flat_cfg or {}) if LOGGERS_AVAILABLE else None,
            )
            fit_start_time = time.time()
            engine.fit(self.model, self.datamodule)
            test_start_time = time.time()
            test_results = engine.test(self.model, self.datamodule)
        job_end_time = time.time()
        durations = {
            "job_duration": job_end_time - job_start_time,
            "fit_duration": test_start_time - fit_start_time,
            "test_duration": job_end_time - test_start_time,
        }
        # TODO(ashwinvaidya17): Restore throughput
        # https://github.com/openvinotoolkit/anomalib/issues/2054
        output = {
            "accelerator": self.accelerator,
            **durations,
            **self.flat_cfg,
            **test_results[0],
        }
        logger.info(f"Completed with result {output}")

        # Logging metrics to External Loggers
        trainer = engine.trainer()
        for logger_instance in trainer.loggers:
            if isinstance(logger_instance, AnomalibCometLogger | AnomalibWandbLogger | AnomalibMLFlowLogger):
                logger_instance.log_metrics(test_results[0])
                logger.debug(f"Successfully logged metrics to {logger_instance.__class__.__name__}")
        return output

    @staticmethod
    def _initialize_loggers(logger_configs: dict[str, dict[str, Any]]) -> list[Any]:
        """Initialize configured external loggers.

        Args:
            logger_configs: Dictionary mapping logger names to their configurations.

        Returns:
            Dictionary of initialized loggers.
        """
        logger_mapping = {
            "tensorboard": AnomalibTensorBoardLogger,
            "comet": AnomalibCometLogger,
            "wandb": AnomalibWandbLogger,
            "mlflow": AnomalibMLFlowLogger,
        }

        active_loggers = []
        default_configs = {
            "tensorboard": {"save_dir": "logs/benchmarks"},
            "comet": {"project_name": "anomalib"},
            "wandb": {"project": "anomalib"},
            "mlflow": {"experiment_name": "anomalib"},
        }

        for logger_name, logger_class in logger_mapping.items():
            # Use provided config or fall back to defaults
            config = logger_configs.get(logger_name, default_configs.get(logger_name, {}))
            logger_instance = logger_class(**config)
            active_loggers.append(logger_instance)
            logger.info(f"Successfully initialized {logger_name} logger")

        return active_loggers

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
            _results = gathered_result.to_dict("list")
            for column in _results:
                table.add_column(column)
            for row in zip(*_results.values(), strict=False):
                table.add_row(*[str(value) for value in row])
            console.print(table)
