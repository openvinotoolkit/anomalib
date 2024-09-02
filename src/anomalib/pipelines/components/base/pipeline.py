"""Base class for pipeline."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from jsonargparse import ArgumentParser, Namespace
from rich import traceback

from anomalib.utils.logging import redirect_logs

from .runner import Runner

if TYPE_CHECKING:
    from anomalib.pipelines.types import PREV_STAGE_RESULT
traceback.install()

log_file = "runs/pipeline.log"
logger = logging.getLogger(__name__)


class Pipeline(ABC):
    """Base class for pipeline."""

    def _get_args(self, args: Namespace) -> dict:
        """Get pipeline arguments by parsing the config file.

        Args:
            args (Namespace): Arguments to run the pipeline. These are the args returned by ArgumentParser.

        Returns:
            dict: Pipeline arguments.
        """
        if args is None:
            logger.warning("No arguments provided, parsing arguments from command line.")
            parser = self.get_parser()
            args = parser.parse_args()

        with Path(args.config).open(encoding="utf-8") as file:
            return yaml.safe_load(file)

    @abstractmethod
    def _setup_runners(self, args: dict) -> list[Runner]:
        """Setup the runners for the pipeline."""

    def run(self, args: Namespace | None = None) -> None:
        """Run the pipeline.

        Args:
            args (Namespace): Arguments to run the pipeline. These are the args returned by ArgumentParser.
        """
        args = self._get_args(args)
        runners = self._setup_runners(args)
        redirect_logs(log_file)
        previous_results: PREV_STAGE_RESULT = None

        for runner in runners:
            try:
                _args = args.get(runner.generator.job_class.name, None)
                previous_results = runner.run(_args, previous_results)
            except Exception:  # noqa: PERF203 catch all exception and allow try-catch in loop
                logger.exception("An error occurred when running the runner.")
                print(
                    f"There were some errors when running {runner.generator.job_class.name} with"
                    f" {runner.__class__.__name__}."
                    f" Please check {log_file} for more details.",
                )

    @staticmethod
    def get_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
        """Create a new parser if none is provided."""
        if parser is None:
            parser = ArgumentParser()
            parser.add_argument("--config", type=str | Path, help="Configuration file path.", required=True)

        return parser
