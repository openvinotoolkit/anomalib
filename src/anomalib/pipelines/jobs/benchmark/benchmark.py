"""Benchmarking job."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import SUPPRESS
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
from jsonargparse import ArgumentParser, Namespace
from jsonargparse._optionals import get_doc_short_description
from lightning import seed_everything
from rich.console import Console
from rich.table import Table

from anomalib.data import AnomalibDataModule, get_datamodule
from anomalib.engine import Engine
from anomalib.models import AnomalyModule, get_model
from anomalib.pipelines.jobs.base import Job
from anomalib.pipelines.utils import (
    dict_from_namespace,
    hide_output,
)
from anomalib.pipelines.utils.actions import GridSearchAction, get_iterator_from_grid_dict


class BenchmarkJob(Job):
    """Benchmarking job."""

    name = "benchmark"

    def __init__(self, accelerator: str) -> None:
        super().__init__()
        self.accelerator = accelerator

    @hide_output
    def run(
        self,
        model: AnomalyModule,
        datamodule: AnomalibDataModule,
        seed: int,
        task_id: int | None = None,
    ) -> dict[str, Any]:
        """Run the benchmark."""
        devices: str | list[int] = "auto"
        if task_id is not None:
            devices = [task_id]
        with TemporaryDirectory() as temp_dir:
            seed_everything(seed)
            engine = Engine(
                accelerator=self.accelerator,
                devices=devices,
                default_root_dir=temp_dir,
            )
            engine.fit(model, datamodule)
            test_results = engine.test(model, datamodule)
        output = {
            "seed": seed,
            "model": model.__class__.__name__,
            "data": datamodule.__class__.__name__,
            "category": datamodule.category,
            **test_results[0],
        }
        self.logger.info(f"Completed with result {output}")
        return output

    def collect(self, results: list[dict[str, Any]]) -> pd.DataFrame:
        """Gather the results returned from run."""
        output: dict[str, Any] = {}
        for key in results[0]:
            output[key] = []
        for result in results:
            for key, value in result.items():
                output[key].append(value)
        return pd.DataFrame(output)

    def save(self, result: pd.DataFrame) -> None:
        """Save the result to a csv file."""
        self._print_tabular_results(result)
        file_path = Path("runs") / self.accelerator / self.name / "results.csv"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(file_path, index=False)
        self.logger.info(f"Saved results to {file_path}")

    def _print_tabular_results(self, gathered_result: pd.DataFrame) -> None:
        """Print the tabular results."""
        if gathered_result is not None:
            console = Console()
            table = Table(title=f"{self.name} Results", show_header=True, header_style="bold magenta")
            _results = gathered_result.to_dict("list")
            for column in _results:
                table.add_column(column)
            for row in zip(*_results.values(), strict=False):
                table.add_row(*[str(value) for value in row])
            console.print(table)

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        """Add job specific arguments to the parser."""
        group = parser.add_argument_group("Benchmark job specific arguments.")
        group.add_argument(
            f"--{BenchmarkJob.name}.seed",
            type=int | dict[str, list[int]],
            default=42,
            help="Seed for reproducibility.",
        )
        BenchmarkJob._add_subclass_arguments(group, AnomalyModule, f"{BenchmarkJob.name}.model")
        BenchmarkJob._add_subclass_arguments(group, AnomalibDataModule, f"{BenchmarkJob.name}.data")

    @staticmethod
    def get_iterator(args: Namespace) -> Iterator:
        """Return iterator based on the arguments."""
        container = {
            "seed": args.seed,
            "data": dict_from_namespace(args.data),
            "model": dict_from_namespace(args.model),
        }
        for _container in get_iterator_from_grid_dict(container):
            yield {
                "seed": _container["seed"],
                "model": get_model(_container["model"]),
                "datamodule": get_datamodule(_container["data"]),
            }

    @staticmethod
    def _add_subclass_arguments(parser: ArgumentParser, baseclass: type, key: str) -> None:
        """Adds the subclass of the provided class to the parser under nested_key."""
        doc_group = get_doc_short_description(baseclass, logger=parser.logger)
        group = parser._create_group_if_requested(  # noqa: SLF001
            baseclass,
            nested_key=key,
            as_group=True,
            doc_group=doc_group,
            config_load=False,
            instantiate=False,
        )

        with GridSearchAction.allow_default_instance_context():
            action = group.add_argument(
                f"--{key}",
                metavar="CONFIG | CLASS_PATH_OR_NAME | .INIT_ARG_NAME VALUE",
                help=(
                    'One or more arguments specifying "class_path"'
                    f' and "init_args" for any subclass of {baseclass.__name__}.'
                ),
                default=SUPPRESS,
                action=GridSearchAction(typehint=baseclass, enable_path=True, logger=parser.logger),
            )
        action.sub_add_kwargs = {"fail_untyped": True, "sub_configs": True, "instantiate": True}
