"""Benchmarking job."""

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
from anomalib.models import AnomalyModule
from anomalib.pipelines.components import Job
from anomalib.utils.logging import hide_output

logger = logging.getLogger(__name__)

# Import external loggers
AVAILABLE_LOGGERS: dict[str, Any] = {}

try:
    from anomalib.loggers import AnomalibCometLogger

    AVAILABLE_LOGGERS["comet"] = AnomalibCometLogger
except ImportError:
    logger.debug("Comet logger not available. Install using `pip install comet-ml`")
try:
    from anomalib.loggers import AnomalibMLFlowLogger

    AVAILABLE_LOGGERS["mlflow"] = AnomalibMLFlowLogger
except ImportError:
    logger.debug("MLflow logger not available. Install using `pip install mlflow`")
try:
    from anomalib.loggers import AnomalibTensorBoardLogger

    AVAILABLE_LOGGERS["tensorboard"] = AnomalibTensorBoardLogger
except ImportError:
    logger.debug("TensorBoard logger not available. Install using `pip install tensorboard`")
try:
    from anomalib.loggers import AnomalibWandbLogger

    AVAILABLE_LOGGERS["wandb"] = AnomalibWandbLogger
except ImportError:
    logger.debug("Weights & Biases logger not available. Install using `pip install wandb`")

LOGGERS_AVAILABLE = len(AVAILABLE_LOGGERS) > 0

if LOGGERS_AVAILABLE:
    logger.info(f"Available loggers: {', '.join(AVAILABLE_LOGGERS.keys())}")
else:
    logger.warning("No external loggers available. Install required packages using `anomalib install -v`")


class BenchmarkJob(Job):
    """Benchmarking job.

    Args:
        accelerator (str): The accelerator to use.
        model (AnomalyModule): The model to use.
        datamodule (AnomalibDataModule): The data module to use.
        seed (int): The seed to use.
        flat_cfg (dict): The flat dictionary of configs with dotted keys.
    """

    name = "benchmark"

    def __init__(
        self,
        accelerator: str,
        model: AnomalyModule,
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
        """Run the benchmark."""
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
                logger=self._initialize_loggers(self.flat_cfg or {}) if LOGGERS_AVAILABLE else [],
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
        # Logging metrics to External Loggers (excluding TensorBoard)
        trainer = engine.trainer()
        for logger_instance in trainer.loggers:
            if any(
                isinstance(logger_instance, AVAILABLE_LOGGERS.get(name, object))
                for name in ["comet", "wandb", "mlflow"]
            ):
                logger_instance.log_metrics(test_results[0])
                logger.debug(f"Successfully logged metrics to {logger_instance.__class__.__name__}")
        return output

    @staticmethod
    def _initialize_loggers(logger_configs: dict[str, dict[str, Any]]) -> list[Any]:
        """Initialize configured external loggers.

        Args:
            logger_configs: Dictionary mapping logger names to their configurations.

        Returns:
            List of initialized loggers.
        """
        active_loggers = []
        default_configs = {
            "tensorboard": {"save_dir": "logs/benchmarks"},
            "comet": {"project_name": "anomalib"},
            "wandb": {"project": "anomalib"},
            "mlflow": {"experiment_name": "anomalib"},
        }

        for logger_name, logger_class in AVAILABLE_LOGGERS.items():
            # Use provided config or fall back to defaults
            config = logger_configs.get(logger_name, default_configs.get(logger_name, {}))
            logger_instance = logger_class(**config)
            active_loggers.append(logger_instance)
            logger.info(f"Successfully initialized {logger_name} logger")

        return active_loggers

    @staticmethod
    def collect(results: list[dict[str, Any]]) -> pd.DataFrame:
        """Gather the results returned from run."""
        output: dict[str, Any] = {}
        for key in results[0]:
            output[key] = []
        for result in results:
            for key, value in result.items():
                output[key].append(value)
        return pd.DataFrame(output)

    @staticmethod
    def save(result: pd.DataFrame) -> None:
        """Save the result to a csv file."""
        BenchmarkJob._print_tabular_results(result)
        file_path = Path("runs") / BenchmarkJob.name / datetime.now().strftime("%Y-%m-%d-%H_%M_%S") / "results.csv"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(file_path, index=False)
        logger.info(f"Saved results to {file_path}")

    @staticmethod
    def _print_tabular_results(gathered_result: pd.DataFrame) -> None:
        """Print the tabular results."""
        if gathered_result is not None:
            console = Console()
            table = Table(title=f"{BenchmarkJob.name} Results", show_header=True, header_style="bold magenta")
            _results = gathered_result.to_dict("list")
            for column in _results:
                table.add_column(column)
            for row in zip(*_results.values(), strict=False):
                table.add_row(*[str(value) for value in row])
            console.print(table)
