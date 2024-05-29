"""Benchmarking job."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
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


class BenchmarkJob(Job):
    """Benchmarking job.

    Args:
        accelerator (str): The accelerator to use.
        model (AnomalyModule): The model to use.
        datamodule (AnomalibDataModule): The data module to use.
        seed (int): The seed to use.
    """

    name = "benchmark"

    def __init__(self, accelerator: str, model: AnomalyModule, datamodule: AnomalibDataModule, seed: int) -> None:
        super().__init__()
        self.accelerator = accelerator
        self.model = model
        self.datamodule = datamodule
        self.seed = seed

    @hide_output
    def run(
        self,
        task_id: int | None = None,
    ) -> dict[str, Any]:
        """Run the benchmark."""
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
            engine.fit(self.model, self.datamodule)
            test_results = engine.test(self.model, self.datamodule)
        # TODO(ashwinvaidya17): Restore throughput
        # https://github.com/openvinotoolkit/anomalib/issues/2054
        output = {
            "seed": self.seed,
            "accelerator": self.accelerator,
            "model": self.model.__class__.__name__,
            "data": self.datamodule.__class__.__name__,
            "category": self.datamodule.category,
            **test_results[0],
        }
        logger.info(f"Completed with result {output}")
        return output

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
