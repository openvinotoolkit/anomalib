"""Benchmark job generator."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from argparse import SUPPRESS
from collections.abc import Generator

from jsonargparse import ArgumentParser, Namespace
from jsonargparse._optionals import get_doc_short_description

from anomalib.data import AnomalibDataModule, get_datamodule
from anomalib.models import AnomalyModule, get_model
from anomalib.pipelines.components import JobGenerator, dict_from_namespace, hide_output
from anomalib.pipelines.components.actions import GridSearchAction, get_iterator_from_grid_dict

from .job import BenchmarkJob


class BenchmarkJobGenerator(JobGenerator):
    """Generate BenchmarkJob."""

    def __init__(self, accelerator: str) -> None:
        self.accelerator = accelerator

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return BenchmarkJob

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
        BenchmarkJobGenerator._add_subclass_arguments(group, AnomalyModule, f"{BenchmarkJob.name}.model")
        BenchmarkJobGenerator._add_subclass_arguments(group, AnomalibDataModule, f"{BenchmarkJob.name}.data")

    @hide_output
    def generate_jobs(self, args: Namespace) -> Generator[BenchmarkJob, None, None]:
        """Return iterator based on the arguments."""
        container = {
            "seed": args.seed,
            "data": dict_from_namespace(args.data),
            "model": dict_from_namespace(args.model),
        }
        for _container in get_iterator_from_grid_dict(container):
            yield BenchmarkJob(
                accelerator=self.accelerator,
                seed=_container["seed"],
                model=get_model(_container["model"]),
                datamodule=get_datamodule(_container["data"]),
            )

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
