"""Benchmarking job."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
from lightning import seed_everything

from anomalib.data import get_datamodule
from anomalib.engine import Engine
from anomalib.models import get_model
from anomalib.pipelines.jobs.base import Job
from anomalib.pipelines.utils import hide_output

from .parser import _Parser


class BenchmarkJob(Job):
    """Benchmarking job."""

    def __init__(self) -> None:
        super().__init__(_Parser(), "benchmark")

    @hide_output
    def run(self, config: dict, task_id: int | None = None) -> dict[str, Any]:  # type: ignore[override]
        """Run the benchmark."""
        self.logger.info(f"Running benchmark with config {config}")
        devices: str | list[int] = "auto"
        if task_id is not None:
            devices = [task_id]
        with TemporaryDirectory() as temp_dir:
            seed_everything(config["seed"])
            model = get_model(config["model"])
            engine = Engine(
                accelerator=config["hardware"],
                devices=devices,
                default_root_dir=temp_dir,
            )
            datamodule = get_datamodule(config)
            engine.fit(model, datamodule)
            test_results = engine.test(model, datamodule)
        output = {
            "seed": config["seed"],
            "model": config["model"]["class_path"].split(".")[-1],
            "data": config["data"]["class_path"].split(".")[-1],
            "category": config["data"]["init_args"].get("category", ""),
            **test_results[0],
        }
        self.logger.info(f"Completed with result {output}")
        return output

    def gather_results(self, results: list[Any]) -> pd.DataFrame:
        """Gather the results returned from run."""
        output: dict[str, Any] = {}
        for key in results[0]:
            output[key] = []
        for result in results:
            for key, value in result.items():
                output[key].append(value)
        return pd.DataFrame(output)

    def tabulate_results(self, results: pd.DataFrame) -> dict[str, Any]:
        """Return the results as a dict."""
        return results.to_dict("list")

    def save_gathered_result(self, result: pd.DataFrame) -> None:
        """Save the result to a csv file."""
        file_path = Path("runs") / self.name / "results.csv"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(file_path, index=False)
        self.logger.info(f"Saved results to {file_path}")
