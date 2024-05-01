"""Base class for pipeline."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod

from jsonargparse import ActionConfigFile, ArgumentParser, Namespace
from rich import print, traceback

from anomalib.utils.logging import redirect_logs

from .runner import Runner

traceback.install()

log_file = "runs/pipeline.log"
logger = logging.getLogger(__name__)


class Pipeline(ABC):
    """Base class for pipeline."""

    def _get_args(self, args: Namespace) -> Namespace:
        """Setup the runners for the pipeline."""
        if args is None:
            logger.warning("No arguments provided, parsing arguments from command line.")
            parser = self.get_parser()
            args = parser.parse_args()
        return args

    @abstractmethod
    def _setup_runners(self, args: Namespace) -> list[Runner]:
        """Setup the runners for the pipeline."""

    def run(self, args: Namespace | None = None) -> None:
        """Run the pipeline.

        Args:
            args (Namespace): Arguments to run the pipeline. These are the args returned by ArgumentParser.
        """
        args = self._get_args(args)
        runners = self._setup_runners(args)
        redirect_logs(log_file)

        for runner in runners:
            try:
                _args = args.get(runner.generator.job_class.name, None)
                runner.run(_args)
            except Exception:  # noqa: PERF203 catch all exception and allow try-catch in loop
                logger.exception("An error occurred when running the runner.")
                print(
                    f"There were some errors when running [red]{runner.generator.job_class.name}[/red] with"
                    f" [green]{runner.__class__.__name__}[/green]."
                    f" Please check [magenta]{log_file}[/magenta] for more details.",
                )

    def get_parser(self, parser: ArgumentParser | None = None) -> ArgumentParser:
        """Create a new parser if none is provided."""
        if parser is None:
            parser = ArgumentParser()
            parser.add_argument("--config", action=ActionConfigFile, help="Configuration file path.")

        return parser
